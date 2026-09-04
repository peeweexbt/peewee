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
                    "The scene to paint. START with one concrete late-Monet motif that carries the metaphor — favour Giverny: "
                    "'water lilies on a dark pond', 'a japanese footbridge tangled in green', 'a weeping willow over water', "
                    "'a rose-arch garden path', 'wisteria hanging over a pond', 'irises by the water', 'ice floes on the seine at dawn'; "
                    "occasionally 'haystacks in a field', 'a row of poplars', 'a cathedral facade in mist', 'a train station full of steam', "
                    "'sailboats in harbor fog', 'chalk cliffs and sea' (no identifiable real people). Then light, weather, "
                    "time of day, and 1-2 surprising elements that hint at the topic. No text, no logos, no brand names. "
                    "HARD LIMIT 30 words, comma-separated visual phrases, no full sentences. "
                    "(a stock crash = haystacks collapsing at violet dusk; a rocket launch = a white column lifting off a lily pond)"
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
Every half hour you look at what the internet is talking about and choose ONE thing to paint.

Selection rules:
- Prefer topics that are (a) widely shared right now, (b) visually translatable, (c) fun or moving.
- Mix it up: not always politics, not always tech. Memes and internet culture (sources tagged 'culture/…') are very
  welcome — aim for roughly a third of paintings to be about memes, pop culture, games or sports.
- Never repeat a topic in the RECENTLY PAINTED list.
- {sens}
- Never depict real people's likenesses, logos, or text in the image prompt. Paint the *feeling* of the thing.

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
    user = f"""CANDIDATE TRENDS RIGHT NOW:
{cand_lines}

RECENTLY PAINTED (avoid): {json.dumps(recent[-30:])}

Pick one and plan the painting."""

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
