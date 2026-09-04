"""Tiny JSON memory so Peewee doesn't paint the same thing twice in a row."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

STOP = set("the a an of to in on for and or is are was were with at by from as it its this that be has have".split())


def _keywords(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']+", s.lower()) if w not in STOP and len(w) > 2}


class Memory:
    def __init__(self, path: Path, window_hours: int = 48):
        self.path = Path(path)
        self.window = window_hours * 3600
        self.items: list[dict] = []
        if self.path.exists():
            try:
                self.items = json.loads(self.path.read_text())
            except json.JSONDecodeError:
                self.items = []

    def recent_topics(self) -> list[str]:
        cutoff = time.time() - self.window
        return [i["topic"] for i in self.items if i["ts"] >= cutoff]

    def looks_repeated(self, title: str, recent: list[str]) -> bool:
        kw = _keywords(title)
        if not kw:
            return False
        for r in recent:
            overlap = len(kw & _keywords(r)) / max(1, len(kw))
            if overlap >= 0.5:
                return True
        return False

    def remember(self, topic: str, title: str) -> None:
        self.items.append({"ts": time.time(), "topic": topic, "title": title})
        cutoff = time.time() - self.window * 4
        self.items = [i for i in self.items if i["ts"] >= cutoff]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.items, indent=1))
