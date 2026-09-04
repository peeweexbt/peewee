"""Peewee's day: look at the internet → pick a thing → paint it → post it. Repeat every 30 min.

Usage:
  python -m peewee.main once            # one painting, then exit (great for cron / GitHub Actions)
  python -m peewee.main loop            # run forever on the schedule in config.yaml
  python -m peewee.main once --dry      # no X post, no git push
  python -m peewee.main once --mock     # no GPU, procedural painting (pipeline test)
  python -m peewee.main plan            # just print what Peewee *would* paint
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

from . import config as cfgmod
from . import brain, painter, publish_site, publish_x, trends
from .memory import Memory

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)-16s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("peewee")


def in_quiet_hours(cfg) -> bool:
    qh = cfg.schedule.quiet_hours
    if not qh or len(qh) != 2:
        return False
    now = datetime.now().strftime("%H:%M")
    a, b = qh
    return (a <= now < b) if a < b else (now >= a or now < b)


def run_once(cfg, dry: bool = False, plan_only: bool = False) -> dict | None:
    root = Path(cfg._root)
    memory = Memory(root / "data" / "memory.json", cfg.trends.avoid_repeat_hours)

    found = trends.gather(cfg)
    plan = brain.think(cfg, found, memory)
    log.info("PLAN  topic=%r title=%r mood=%r", plan.topic, plan.title, plan.mood)
    log.info("PROMPT %s", painter.full_prompt(cfg, plan.image_prompt, plan.colour_words))
    log.info("TWEET  %s", plan.tweet)
    if plan_only:
        return plan.to_dict()

    pid = datetime.now().strftime("%Y%m%d-%H%M") + "-" + uuid.uuid4().hex[:4]
    img_path = root / cfg.site.gallery_dir / "paintings" / f"{pid}.jpg"
    painter.paint(cfg, plan, img_path)

    entry = publish_site.add_to_feed(cfg, plan, img_path, pid)
    memory.remember(plan.topic, plan.title)

    page_url = f"{cfg.site.base_url.rstrip('/')}/#{pid}" if cfg.site.base_url else None
    if dry:
        log.info("dry run: skipping X post + git push")
    else:
        try:
            publish_x.post(cfg, plan, img_path, page_url)
        except Exception as e:  # noqa: BLE001
            log.error("X post failed: %s", e)
        publish_site.git_publish(cfg, f"peewee: {plan.title} ({plan.topic})")
    log.info("done → %s", img_path)
    return entry


def loop(cfg, dry: bool = False) -> None:
    every = int(cfg.schedule.every_minutes) * 60
    jitter = int(cfg.schedule.jitter_minutes) * 60
    log.info("Peewee is awake. painting every %d min.", every // 60)
    while True:
        started = time.time()
        if in_quiet_hours(cfg):
            log.info("quiet hours — Peewee is napping")
        else:
            try:
                run_once(cfg, dry=dry)
            except Exception as e:  # noqa: BLE001
                log.exception("cycle failed: %s", e)
        wait = max(60, every - (time.time() - started) + random.uniform(-jitter, jitter))
        log.info("next painting in %.0f min", wait / 60)
        time.sleep(wait)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="peewee")
    ap.add_argument("mode", choices=["once", "loop", "plan"])
    ap.add_argument("--dry", action="store_true", help="don't tweet or git push")
    ap.add_argument("--mock", action="store_true", help="use the procedural mock painter")
    ap.add_argument("--config", default=None)
    a = ap.parse_args(argv)

    cfg = cfgmod.load(a.config)
    if a.mock:
        cfg["painter"]["backend"] = "mock"
    if a.mode == "plan":
        run_once(cfg, plan_only=True)
    elif a.mode == "once":
        run_once(cfg, dry=a.dry)
    else:
        loop(cfg, dry=a.dry)
    return 0


if __name__ == "__main__":
    sys.exit(main())
