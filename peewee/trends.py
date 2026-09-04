"""Gather what the internet is talking about right now.

Every source returns a list of Trend(title, source, url, score, blurb).
Sources are best-effort: any one of them failing never stops Peewee painting.
"""
from __future__ import annotations

import concurrent.futures as cf
import logging
import re
import time
from dataclasses import dataclass, asdict, field
from typing import Iterable

import requests

from .config import env

log = logging.getLogger("peewee.trends")
UA = {"User-Agent": env("REDDIT_USER_AGENT", "peewee-paints/1.0 (art bot; contact via github)")}
TIMEOUT = 12


@dataclass
class Trend:
    title: str
    source: str
    url: str = ""
    score: float = 0.0
    blurb: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ── Reddit ────────────────────────────────────────────────────────────────
def _reddit_token() -> str | None:
    cid, sec = env("REDDIT_CLIENT_ID"), env("REDDIT_CLIENT_SECRET")
    if not (cid and sec):
        return None
    try:
        r = requests.post(
            "https://www.reddit.com/api/v1/access_token",
            auth=(cid, sec),
            data={"grant_type": "client_credentials"},
            headers=UA,
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        return r.json()["access_token"]
    except Exception as e:  # noqa: BLE001
        log.warning("reddit oauth failed (%s); falling back to public json", e)
        return None


def reddit(subreddits: Iterable[str], limit: int = 15) -> list[Trend]:
    token = _reddit_token()
    base = "https://oauth.reddit.com" if token else "https://www.reddit.com"
    headers = dict(UA)
    if token:
        headers["Authorization"] = f"bearer {token}"
    out: list[Trend] = []
    for sub in subreddits:
        try:
            r = requests.get(f"{base}/r/{sub}/hot.json", params={"limit": limit}, headers=headers, timeout=TIMEOUT)
            r.raise_for_status()
            for child in r.json()["data"]["children"]:
                d = child["data"]
                if d.get("stickied") or d.get("over_18"):
                    continue
                out.append(
                    Trend(
                        title=d["title"],
                        source=f"reddit/r/{d.get('subreddit', sub)}",
                        url="https://reddit.com" + d.get("permalink", ""),
                        score=float(d.get("score", 0)),
                        blurb=(d.get("selftext") or "")[:300],
                        tags=[d.get("subreddit", sub).lower()],
                    )
                )
            time.sleep(0.6)  # be polite to the public endpoint
        except Exception as e:  # noqa: BLE001
            log.warning("reddit r/%s failed: %s", sub, e)
    return out


# ── Google Trends (RSS of daily trending searches) ────────────────────────
def google_trends(geo: str = "US") -> list[Trend]:
    import feedparser

    url = f"https://trends.google.com/trending/rss?geo={geo}"
    try:
        feed = feedparser.parse(requests.get(url, headers=UA, timeout=TIMEOUT).content)
        out = []
        for i, e in enumerate(feed.entries[:25]):
            traffic = e.get("ht_approx_traffic", "0").replace("+", "").replace(",", "")
            try:
                score = float(traffic)
            except ValueError:
                score = max(1.0, 25 - i) * 1000
            blurb = ""
            news = e.get("ht_news_item_title") or ""
            if news:
                blurb = news
            out.append(Trend(title=e.title, source="google-trends", url=e.get("link", ""), score=score, blurb=blurb, tags=["search"]))
        return out
    except Exception as e:  # noqa: BLE001
        log.warning("google trends failed: %s", e)
        return []


# ── News RSS ──────────────────────────────────────────────────────────────
def news_rss(feeds: Iterable[str], per_feed: int = 12, label: str = "news") -> list[Trend]:
    import feedparser

    out: list[Trend] = []
    for f in feeds:
        try:
            feed = feedparser.parse(requests.get(f, headers=UA, timeout=TIMEOUT).content)
            host = re.sub(r"^www\.|^feeds\.|^rss\.", "", requests.utils.urlparse(f).netloc)
            for i, e in enumerate(feed.entries[:per_feed]):
                summary = re.sub(r"<[^>]+>", "", e.get("summary", ""))[:300]
                out.append(Trend(title=e.title, source=f"{label}/{host}", url=e.get("link", ""), score=float(per_feed - i), blurb=summary, tags=[label]))
        except Exception as e:  # noqa: BLE001
            log.warning("rss %s failed: %s", f, e)
    return out


# ── Hacker News ───────────────────────────────────────────────────────────
def hackernews(limit: int = 20) -> list[Trend]:
    try:
        ids = requests.get("https://hacker-news.firebaseio.com/v0/topstories.json", timeout=TIMEOUT).json()[:limit]
        out = []
        for i in ids:
            it = requests.get(f"https://hacker-news.firebaseio.com/v0/item/{i}.json", timeout=TIMEOUT).json() or {}
            if it.get("title"):
                out.append(Trend(title=it["title"], source="hackernews", url=it.get("url", f"https://news.ycombinator.com/item?id={i}"), score=float(it.get("score", 0)), tags=["tech"]))
        return out
    except Exception as e:  # noqa: BLE001
        log.warning("hackernews failed: %s", e)
        return []


# ── X trends (paid tier only) ─────────────────────────────────────────────
def x_trends(woeid: int = 23424977) -> list[Trend]:
    bearer = env("X_BEARER_TOKEN")
    if not bearer:
        return []
    try:
        r = requests.get(
            f"https://api.x.com/2/trends/by/woeid/{woeid}",
            headers={"Authorization": f"Bearer {bearer}"},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        out = []
        for i, t in enumerate(r.json().get("data", [])[:30]):
            out.append(Trend(title=t.get("trend_name", ""), source="x-trends", score=float(t.get("tweet_count") or (30 - i) * 1000), tags=["x"]))
        return out
    except Exception as e:  # noqa: BLE001
        log.warning("x trends failed: %s", e)
        return []


# ── Aggregate ─────────────────────────────────────────────────────────────
def _normalise(trends: list[Trend]) -> list[Trend]:
    """Scale scores per-source to 0..1 so Reddit's 50k upvotes don't drown RSS."""
    by_src: dict[str, list[Trend]] = {}
    for t in trends:
        by_src.setdefault(t.source.split("/")[0], []).append(t)
    for group in by_src.values():
        mx = max((t.score for t in group), default=1.0) or 1.0
        for t in group:
            t.score = round(t.score / mx, 3)
    return trends


def gather(cfg) -> list[Trend]:
    tc = cfg.trends
    jobs = []
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        if tc.reddit.enabled:
            jobs.append(ex.submit(reddit, tc.reddit.subreddits, tc.reddit.limit_per_sub))
        if tc.google_trends.enabled:
            jobs.append(ex.submit(google_trends, tc.google_trends.geo))
        if tc.news_rss.enabled:
            jobs.append(ex.submit(news_rss, tc.news_rss.feeds))
        if tc.get("culture_rss", {}).get("enabled"):
            jobs.append(ex.submit(news_rss, tc.culture_rss.feeds, 10, "culture"))
        if tc.hackernews.enabled:
            jobs.append(ex.submit(hackernews))
        if tc.x_trends.enabled:
            jobs.append(ex.submit(x_trends, tc.x_trends.woeid))
        results: list[Trend] = []
        for j in cf.as_completed(jobs):
            try:
                results.extend(j.result())
            except Exception as e:  # noqa: BLE001
                log.warning("trend job failed: %s", e)
    results = [t for t in results if t.title and len(t.title) > 6]
    log.info("gathered %d trend candidates", len(results))
    return _normalise(results)
