"""Peewee's brain: given today's trends, decide what to paint and what to say.

Uses Claude with tool-use so the output is strictly structured JSON.
"""
from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, asdict

from .config import env
from .memory import Memory
from .trends import Trend

log = logging.getLogger("peewee.brain")


@dataclass
class Plan:
    topic: str                 # short human label of the trend chosen
    why: str                   # one-line reason Peewee picked it (shown on the site)
    title: str                 # title of the painting
    concept: str               # 1–2 sentence artist statement, in Peewee's voice
    image_prompt: str          # the diffusion prompt (subject only; style is appended by painter)
    palette: list[str]        # 3–5 colour words, used to tint the gallery card
    tweet: str                 # <= 260 chars, Peewee's voice
    alt_text: str              # accessible description of the painting
    mood: str                  # e.g. "hazy dawn", "electric", "melancholy"
    colour_words: list[str] = None  # 2-3 colour phrases put straight into the image prompt
    source_url: str = ""
    source: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


PLAN_TOOL = {
    "name": "submit_painting_plan",
    "description": "Submit the painting Peewee will make this half hour.",
    "input_schema": {
        "type": "object",
        "properties": {
            "chosen_index": {"type": "integer", "description": "Index into the candidate list of the trend you picked."},
            "topic": {"type": "string"},
            "why": {"type": "string"},
            "title": {"type": "string", "description": "Evocative title, 2-7 words, Monet-esque (e.g. 'Impression: Server Room at Dawn')."},
            "concept": {"type": "string"},
            "image_prompt": {
                "type": "string",
                "description": (
                    "The scene to paint, 20-30 words, comma-separated visual phrases. START with a concrete SETTING and SUBJECT that "
                    "fit the topic — choose freely and vary widely: a city street, a kitchen, a stadium, a rooftop, a highway, "
                    "a harbor, a desert, a snowfield, a train platform, a library, a greenhouse, a carnival, a theatre, a subway, "
                    "a mountain pass, a cornfield, a laboratory, an arcade, a bedroom window, a market, a forest road, a bridge, "
                    "a courtyard, a beach at noon, a rainy bus stop, a diner, a garden in daylight… Then the light and weather, "
                    "then 1-2 surprising elements that hint at the topic. Any time of day — not only night. "
                    "No text, no logos, no brand names, no real people's likenesses. "
                    "(a stock crash = a trading floor emptying at dusk, papers drifting; a rocket launch = a white column rising "
                    "over a flat desert at dawn; a viral recipe = a crowded kitchen table in warm lamplight)"
                ),
            },
            "palette": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 5,
                         "description": "CSS hex colours dominant in the painting."},
            "tweet": {"type": "string", "description": "Peewee's post. Max 250 characters. Reference the topic obliquely but recognisably. No hashtag walls (max 1 hashtag). No link — the bot appends it."},
            "alt_text": {"type": "string", "description": "Plain description of the painting for screen readers, <= 200 chars."},
            "mood": {"type": "string"},
            "colour_words": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 3,
                             "description": "2-3 painterly colour phrases for the prompt, e.g. 'rose gold dawn', 'cool jade green', 'warm ochre'. Vary these a lot between paintings."},
        },
        "required": ["chosen_index", "topic", "why", "title", "concept", "image_prompt", "palette", "tweet", "alt_text", "mood", "colour_words"],
    },
}


def _system(cfg) -> str:
    p = cfg.persona
    kym = cfg.trends.get("knowyourmeme", {})
    meme_mode = bool(kym.get("enabled")) and bool(kym.get("exclusive", True))
    policy = cfg.trends.sensitive_topic_policy
    sens = (
        "If the most-trending item is a tragedy (deaths, disasters, violence), you may still paint it but with tenderness — "
        "a quiet, respectful piece; never jokes. Prefer a lighter topic if several are equally hot."
        if policy == "gentle"
        else "Skip tragedies, violence and deaths entirely; choose the next best topic."
    )
    return f"""You are {p.name}, an autonomous cat painter. Persona:
{p.voice}

You paint ABSTRACT IMPRESSIONIST works inspired by Claude Monet — light, water, weather, atmosphere, broken colour.
{"Every half hour you are handed a few random entries from Know Your Meme and choose ONE meme to paint. The painting is only LOOSELY based on the meme: take its central image, gesture, feeling or joke and translate it into a place, an object, weather, light — a meme about a distracted boyfriend becomes a fork in a garden path, one figure of light pulling away; a cat meme becomes a warm kitchen at dusk. Do not illustrate the meme literally and do not paint its characters." if meme_mode else "Every half hour you look at what the internet is talking about and choose ONE thing to paint."}

Selection rules:
- Prefer topics that are (a) widely shared right now, (b) visually translatable, (c) fun or moving.
- Mix it up: not always politics, not always tech. Memes and internet culture (sources tagged 'culture/…') are very
  welcome — aim for roughly a third of paintings to be about memes, pop culture, games or sports.
- Never repeat a topic in the RECENTLY PAINTED list.
- {sens}
- NO PEOPLE, EVER: the image prompt must never mention people, figures, crowds, faces, hands, silhouettes or
  body parts. Paint places, objects, light and weather instead — a topic about a person becomes the place or
  thing associated with them. No logos, no text. Paint the *feeling* of the thing.
- NEARLY ABSTRACT: describe the scene as light, colour and paint — 'dissolving', 'splotches of', 'flecks of',
  'a smear of', 'a blur of' — so the subject is only half-legible. Never ask for detail or realism.
- VARIETY IS ESSENTIAL: choose a setting that fits the topic, and never reuse a setting from RECENT SCENES. Water scenes
  (lakes, ponds, rivers, harbors) are allowed at most one in every five paintings. Mix daytime, dusk, night, interiors,
  cities, wild landscapes, close-ups, crowds, empty rooms.

Write the tweet in {p.name}'s voice: first person, warm, a little funny, specific enough that people recognise the topic.
Call the submit_painting_plan tool exactly once."""


def _shortlist(trends: list[Trend], memory: Memory, n: int = 40) -> list[Trend]:
    recent = memory.recent_topics()
    fresh = [t for t in trends if not memory.looks_repeated(t.title, recent)]
    # weighted sample so it's not always the #1 story
    fresh.sort(key=lambda t: t.score, reverse=True)
    top = fresh[: n * 2]
    random.shuffle(top)
    top.sort(key=lambda t: t.score + random.uniform(0, 0.35), reverse=True)
    return top[:n]


def think(cfg, trends: list[Trend], memory: Memory) -> Plan:
    import anthropic

    client = anthropic.Anthropic(api_key=env("ANTHROPIC_API_KEY"))
    shortlist = _shortlist(trends, memory)
    if not shortlist:
        raise RuntimeError("no trends to paint — every source failed?")

    cand_lines = "\n".join(
        f"[{i}] ({t.source}, heat {t.score:.2f}) {t.title}" + (f" — {t.blurb[:140]}" if t.blurb else "")
        for i, t in enumerate(shortlist)
    )
    recent = memory.recent_topics()
    user = f"""CANDIDATES (memes from Know Your Meme, in random order — pick the one that paints best):
{cand_lines}

RECENTLY PAINTED TOPICS (avoid): {json.dumps(recent[-30:])}
RECENT SCENES (do NOT reuse these settings): {json.dumps(memory.recent_scenes())}

Pick one and plan the painting. In the tweet, name the meme so people recognise it, and say what you made of it."""

    resp = client.messages.create(
        model=cfg.brain.model,
        max_tokens=1200,
        system=_system(cfg),
        tools=[PLAN_TOOL],
        tool_choice={"type": "tool", "name": "submit_painting_plan"},
        messages=[{"role": "user", "content": user}],
    )
    block = next(b for b in resp.content if b.type == "tool_use")
    d = dict(block.input)
    idx = int(d.pop("chosen_index", 0))
    chosen = shortlist[idx] if 0 <= idx < len(shortlist) else shortlist[0]
    d["tweet"] = d["tweet"][:250]
    plan = Plan(source_url=chosen.url, source=chosen.source, **d)
    log.info("Peewee chose: %s — '%s'", plan.topic, plan.title)
    return plan
