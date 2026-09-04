"""Download high-res, public-domain Monet paintings for LoRA training.

Sources (both free, no API key):
  • Art Institute of Chicago open-access API (CC0 images, IIIF)
  • Wikimedia Commons (search, filtered to large JPEGs)

python train/fetch_monet.py                  # late period (1897+) — Giverny, water lilies, willows, bridge
python train/fetch_monet.py --from-year 1870 # everything
python train/fetch_monet.py --max 80

Files land in train/data/raw/. Then run prepare_dataset.py as usual.
Always eyeball the folder afterwards and delete anything that isn't a painting (frames, gallery photos, sketches).
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import requests

UA = {"User-Agent": "peewee-paints-trainer/1.0 (personal art bot; downloads public-domain Monet)"}
OUT = Path("train/data/raw")
LATE_TERMS = ["water lilies", "nymphéas", "nympheas", "japanese bridge", "japanese footbridge", "weeping willow",
              "wisteria", "rose", "giverny", "irises", "agapanthus", "path", "ice floes", "morning on the seine",
              "house seen from the rose garden", "flowering arches"]


def safe(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:70]


def save(url: str, name: str, min_px: int) -> bool:
    try:
        r = requests.get(url, headers=UA, timeout=60)
        r.raise_for_status()
        if len(r.content) < 80_000:
            return False
        from PIL import Image
        import io
        im = Image.open(io.BytesIO(r.content))
        if min(im.size) < min_px:
            return False
        (OUT / f"{name}.jpg").write_bytes(r.content)
        print(f"  ✓ {name}  {im.size[0]}×{im.size[1]}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  ✗ {name}: {e}")
        return False


def artic(from_year: int, max_n: int, min_px: int) -> int:
    print("Art Institute of Chicago…")
    n = 0
    page = 1
    while n < max_n and page <= 6:
        r = requests.get(
            "https://api.artic.edu/api/v1/artworks/search",
            params={
                "q": "Claude Monet",
                "query[term][is_public_domain]": "true",
                "fields": "id,title,image_id,date_end,date_start,artist_title",
                "limit": 50,
                "page": page,
            },
            headers=UA, timeout=30,
        ).json()
        data = r.get("data", [])
        if not data:
            break
        for a in data:
            if a.get("artist_title") != "Claude Monet" or not a.get("image_id"):
                continue
            if (a.get("date_end") or 0) < from_year:
                continue
            url = f"https://www.artic.edu/iiif/2/{a['image_id']}/full/1686,/0/default.jpg"
            if save(url, f"artic-{a['id']}-{safe(a['title'])}", min_px):
                n += 1
                if n >= max_n:
                    break
            time.sleep(0.4)
        page += 1
    return n


def commons(terms: list[str], max_n: int, min_px: int) -> int:
    print("Wikimedia Commons…")
    api = "https://commons.wikimedia.org/w/api.php"
    seen: set[str] = set()
    n = 0
    for term in terms:
        if n >= max_n:
            break
        q = f'"Claude Monet" {term}'
        r = requests.get(api, params={"action": "query", "list": "search", "srsearch": q, "srnamespace": 6,
                                      "srlimit": 30, "format": "json"}, headers=UA, timeout=30).json()
        titles = [h["title"] for h in r.get("query", {}).get("search", []) if h["title"].lower().endswith((".jpg", ".jpeg"))]
        titles = [t for t in titles if t not in seen and "monet" in t.lower()]
        if not titles:
            continue
        info = requests.get(api, params={"action": "query", "titles": "|".join(titles[:20]), "prop": "imageinfo",
                                         "iiprop": "url|size|extmetadata", "iiurlwidth": 1600, "format": "json"},
                            headers=UA, timeout=30).json()
        for pg in info.get("query", {}).get("pages", {}).values():
            ii = (pg.get("imageinfo") or [{}])[0]
            if not ii or ii.get("width", 0) < min_px:
                continue
            meta = ii.get("extmetadata", {})
            artist = (meta.get("Artist", {}).get("value") or "").lower()
            if artist and "monet" not in artist:
                continue
            seen.add(pg["title"])
            url = ii.get("thumburl") or ii["url"]
            if save(url, f"commons-{safe(pg['title'][5:])}", min_px):
                n += 1
                if n >= max_n:
                    break
            time.sleep(0.4)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-year", type=int, default=1897, help="earliest year to include (1897 = the Giverny / water-lily era)")
    ap.add_argument("--max", type=int, default=60, help="target number of images in total")
    ap.add_argument("--min-px", type=int, default=900, help="skip images whose short side is smaller than this")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    got = artic(a.from_year, a.max, a.min_px)
    if got < a.max:
        got += commons(LATE_TERMS if a.from_year >= 1890 else LATE_TERMS + ["haystacks", "poplars", "rouen cathedral", "argenteuil", "etretat"], a.max - got, a.min_px)
    print(f"\n{got} images in {OUT}. Open the folder, delete anything that isn't a painting, then:")
    print("  python train/prepare_dataset.py --trigger pwstyle")


if __name__ == "__main__":
    main()
