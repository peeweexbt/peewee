"""Append the new painting to gallery/feed.json and (optionally) git-push the gallery."""
from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("peewee.site")


def add_to_feed(cfg, plan, image_path: Path, painting_id: str) -> dict:
    root = Path(cfg._root)
    gdir = root / cfg.site.gallery_dir
    feed_path = gdir / "feed.json"
    feed = {"title": cfg.site.title, "tagline": cfg.site.tagline, "items": []}
    if feed_path.exists():
        try:
            feed = json.loads(feed_path.read_text())
        except json.JSONDecodeError:
            pass
    rel = image_path.relative_to(gdir).as_posix()
    entry = {
        "id": painting_id,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "image": rel,
        "thumb": rel.replace(".jpg", "_thumb.jpg"),
        **plan.to_dict(),
    }
    feed["items"].insert(0, entry)
    feed["items"] = feed["items"][: int(cfg.site.max_items_in_feed)]
    feed["title"], feed["tagline"] = cfg.site.title, cfg.site.tagline
    feed["handle"] = cfg.persona.handle
    feed["updated"] = entry["ts"]
    feed_path.write_text(json.dumps(feed, indent=1, ensure_ascii=False))
    log.info("feed updated (%d items)", len(feed["items"]))
    return entry


def git_publish(cfg, message: str) -> None:
    if cfg.site.publish.method != "git":
        return
    root = cfg._root
    gdir = cfg.site.gallery_dir
    try:
        subprocess.run(["git", "add", gdir, "data/memory.json"], cwd=root, check=True)
        subprocess.run(["git", "commit", "-m", message, "--quiet"], cwd=root, check=True)
        subprocess.run(["git", "push", cfg.site.publish.remote, cfg.site.publish.branch, "--quiet"], cwd=root, check=True)
        log.info("pushed to %s/%s", cfg.site.publish.remote, cfg.site.publish.branch)
    except subprocess.CalledProcessError as e:
        log.error("git publish failed: %s", e)
