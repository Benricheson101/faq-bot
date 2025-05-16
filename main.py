import json
import os
from typing import Tuple
import discord
from sentence_transformers import SentenceTransformer, util
import torch
import toml
from discord import AllowedMentions, Intents, Message
from transformers.pipelines import pipeline
from watchdog.events import PatternMatchingEventHandler
from watchdog.observers import Observer

FAQ_DEFAULT_THRESHOLD = 0.75
QUESTION_THRESHOLD = 0.75

faqs_file = "faq.toml"

faq_model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")
classifier_model = pipeline(
    "zero-shot-classification", model="facebook/bart-large-mnli"
)

run = True
debug_channel = 1372001351516688495


def reload_faqs():
    global faq_questions
    global faq_embeddings
    global faqs
    global faqs_parsed

    faqs_parsed = toml.load(faqs_file)

    faqs = {
        q: {
            "question": q,
            "answer": faq["answer"],
            "threshold": float(faq.get("threshold", FAQ_DEFAULT_THRESHOLD)),
            "require_question": faq.get("question", True),
            "require_question_threshold": faq.get(
                "question_threshold", QUESTION_THRESHOLD
            ),
        }
        for faq in faqs_parsed["faq"]
        for q in faq["questions"]
    }

    faq_questions = [q for q in faqs.keys()]
    faq_embeddings = faq_model.encode(
        faq_questions, convert_to_tensor=True, normalize_embeddings=True
    )


reload_faqs()


class Watcher(PatternMatchingEventHandler):
    def __init__(self):
        super().__init__(patterns=["*.toml"])

    def on_modified(self, event):
        reload_faqs()


def get_faq(question: str) -> Tuple[dict, float]:
    question_embedding = faq_model.encode(
        question, convert_to_tensor=True, normalize_embeddings=True
    )

    dot_scores = util.dot_score(question_embedding, faq_embeddings)

    best_score, best_match_idx = torch.max(dot_scores[0], dim=0)
    best_score = best_score.item()
    best_match_idx = best_match_idx.item()

    return (faqs[faq_questions[best_match_idx]], best_score)  # type: ignore


def is_question(msg: str):
    global classifier_model
    try:
        output = classifier_model(msg, ["question", "statement"])
        return (output["labels"][0] == "question", output["scores"][0])  # type: ignore
    except Exception as e:
        print("classification failed:", e)
        return (False, 1.0)


class Client(discord.Client):
    async def on_ready(self):
        print("ready as", self.user)

    async def on_message(self, msg: Message):
        global faqs_parsed

        channels = faqs_parsed.get("channels", [])
        guilds = faqs_parsed.get("guilds", [])

        if (
            msg.author.bot
            or msg.author == self.user
            or msg.guild is None
            or not (str(msg.guild.id) in guilds if len(guilds) != 0 else True)
            or not (str(msg.channel.id) in channels if len(channels) != 0 else True)
        ):
            return

        global run
        if msg.channel.id == debug_channel:
            match msg.content:
                case "!stop":
                    run = False
                    return
                case "!start":
                    run = True
                    return

        debug = msg.channel.id == debug_channel
        if msg.content.startswith("!debug"):
            msg.content = msg.content[len("!debug ") :]
            debug = True

        if not run:
            return

        if (
            len(msg.content) > 256
            or not msg.content
            or not msg.content[0].isalpha()
            or msg.content.count(" ") < 1
        ):
            return

        resp, score = get_faq(msg.content)

        is_a_question, is_a_question_confidence = is_question(msg.content)
        require_question_threshold = resp.get(
            "require_question_threshold", QUESTION_THRESHOLD
        )

        require_question = resp.get("require_question", True)
        is_question_enough = (not require_question) or (
            is_a_question
            and (
                is_a_question_confidence
                >= resp.get("require_question_threshold", QUESTION_THRESHOLD)
            )
        )

        send = score >= resp["threshold"] and is_question_enough
        if debug:
            await msg.reply(
                "```json\n"
                + json.dumps(
                    {
                        "send": send,
                        "input": msg.content.strip(),
                        "question": {
                            "is_question": is_a_question,
                            "confidence": is_a_question_confidence,
                            "threshold": require_question_threshold,
                            "is_question_enough": is_question_enough,
                        },
                        "faq": {
                            "question": resp["question"],
                            "answer": resp["answer"],
                            "threshold": resp["threshold"],
                            "score": score,
                        },
                    },
                    indent=2,
                )
                + "\n```",
                mention_author=False,
                allowed_mentions=AllowedMentions.none(),
            )
        if not send:
            return

        await msg.reply(
            f"{resp["answer"]}\n-# {int(score * 100)}% confidence",
            mention_author=True,
            allowed_mentions=AllowedMentions.none(),
        )


intents = Intents.default()
intents.message_content = True

client = Client(
    intents=intents,
)

token = os.getenv("DISCORD_TOKEN")
assert token
w = Watcher()
o = Observer()
o.schedule(w, ".", recursive=False)
o.start()
client.run(token)
