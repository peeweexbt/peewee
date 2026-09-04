"""Post the painting to X. Media upload uses v1.1 (still the media endpoint), the tweet uses v2."""
from __future__ import annotations

import logging
from pathlib import Path

from .config import env

log = logging.getLogger("peewee.x")


def _clients():
    import tweepy

    ck, cs, at, ats = env("X_API_KEY"), env("X_API_SECRET"), env("X_ACCESS_TOKEN"), env("X_ACCESS_TOKEN_SECRET")
    if not all((ck, cs, at, ats)):
        raise RuntimeError("X keys missing in .env")
    auth = tweepy.OAuth1UserHandler(ck, cs, at, ats)
    v1 = tweepy.API(auth)
    v2 = tweepy.Client(consumer_key=ck, consumer_secret=cs, access_token=at, access_token_secret=ats)
    return v1, v2


def post(cfg, plan, image_path: Path, page_url: str | None) -> str | None:
    if not cfg.x.enabled:
        log.info("X posting disabled; would have tweeted: %s", plan.tweet)
        return None
    v1, v2 = _clients()
    media = v1.media_upload(filename=str(image_path))
    if cfg.x.alt_text and plan.alt_text:
        try:
            v1.create_media_metadata(media.media_id, plan.alt_text[:1000])
        except Exception as e:  # noqa: BLE001
            log.warning("alt text failed: %s", e)
    text = plan.tweet.strip()
    if cfg.x.include_link and page_url:
        text = f"{text}\n\n{page_url}"
    resp = v2.create_tweet(text=text[:280], media_ids=[media.media_id])
    tid = resp.data["id"]
    log.info("tweeted https://x.com/i/status/%s", tid)
    return tid
