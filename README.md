# Peewee Paints 🐾

Peewee is a small grey tabby who holds a brush in his mouth and paints an **abstract impressionist
painting, inspired by Monet, about whatever the internet is talking about — every 30 minutes** —
then hangs it in his online gallery and posts it on X.

```
 every 30 min ─▶ trends.py   scrape Reddit / Google Trends / news RSS / HN (/ X trends)
              ─▶ brain.py    Claude, in Peewee's voice, picks ONE topic → title, concept,
                             diffusion prompt, tweet, alt-text, palette   (structured tool-call)
              ─▶ painter.py  Stable Diffusion (local GPU / your remote SD box / Replicate)
                             + optional LoRA trained on YOUR images
              ─▶ publish_site.py   writes gallery/feed.json + jpgs, git-pushes → GitHub Pages
              ─▶ publish_x.py      uploads image + tweet via tweepy
```

## Quick start (5 minutes, no GPU, no keys except Claude)

```bash
git clone <this repo> peewee && cd peewee
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-scheduler.txt
cp .env.example .env            # add ANTHROPIC_API_KEY
python -m peewee.main plan      # see what Peewee would paint right now
python -m peewee.main once --dry --mock   # full cycle with a procedural mock painting
open gallery/index.html         # (serve it: python -m http.server -d gallery 8080)
```

## Making real paintings — where does the GPU live?

Stable Diffusion needs a GPU (or Apple Silicon). Your scheduler does not. Pick a layout:

| Layout | How | Cost |
|---|---|---|
| **A. Everything on your Mac** (M1/M2/M3/M4) | `pip install -r requirements.txt`, `painter.backend: local`, run `loop` via `deploy/com.peewee.paints.plist`. SDXL ≈ 60–120 s/image on an M-series. Stops when the Mac sleeps (turn off sleep or use `caffeinate`). | free |
| **B. GPU VPS** (recommended for 24/7) | Rent a small GPU box (RunPod, Lambda, Vast, Hetzner GPU, Paperspace). Install everything, `painter.backend: local`, run as `deploy/peewee.service`. | ≈ $0.20–0.50/h |
| **C. Cheap VPS + your GPU box** | Fly/Railway runs the scheduler (`deploy/Dockerfile`, `PEEWEE_PAINTER=remote_sd`), and it calls an AUTOMATIC1111/Forge/ComfyUI API on any machine you own (expose it with Tailscale or Cloudflare Tunnel). | $5/mo + your box |
| **D. Cheap VPS + Replicate** | Same scheduler, `PEEWEE_PAINTER=replicate`. Zero GPU ops; fine as a fallback. | $5/mo + ~$0.03/image |

`painter.backend` can also be flipped at runtime with the `PEEWEE_PAINTER` env var, so you can
keep `local` in config and fall back to `replicate` from the VPS.

## The website

`gallery/` is a static site: `index.html` + `feed.json` + `paintings/*.jpg`. Nothing to build.

1. Push this repo to GitHub, Settings → Pages → Source: **GitHub Actions**. The included
   `.github/workflows/pages.yml` redeploys every time the bot pushes a painting.
2. Set `site.base_url` in `config.yaml` to your Pages URL (used for the link in tweets).
3. The bot's machine needs push access: an SSH deploy key with write access, or a fine-grained PAT
   in the git remote URL. (`site.publish.method: none` if you'd rather sync some other way, e.g. rsync/S3.)

Vercel/Netlify work too: point them at the repo with the publish directory set to `gallery`.

## X / Twitter

1. developer.x.com → create a project + app for Peewee's account.
2. App settings → **User authentication settings** → App permissions: **Read and write**. Save.
3. Keys & tokens → *regenerate* the Access Token & Secret **after** step 2 (tokens made before it are read-only).
4. Paste API key/secret + access token/secret into `.env`, set `x.enabled: true`.
5. `python -m peewee.main once` — check Peewee's timeline.

Heads-up: X's free tier is very limited and the paid tiers/pricing have changed several times
(they moved toward a credit/usage-based model in 2026). 48 image posts a day is ~1,450/month — check the
current limits on developer.x.com before committing to a plan. `trends.x_trends` needs a tier with
trends access and is off by default; Reddit/Google Trends/news give plenty of signal without it.

## Tuning Peewee

- **Voice & taste**: `persona.voice` and the system prompt in `brain.py`.
- **Style**: `painter.style_prompt` — currently Monet-forward abstract impressionism. Try adding
  "Water Lilies series palette" or "Rouen Cathedral light study" for variety.
- **Cadence**: `schedule.every_minutes`, `quiet_hours` (let him sleep), `jitter_minutes`.
- **Sources**: toggle any block in `trends`; add subreddits or RSS feeds freely.
- **Sensitive news**: `trends.sensitive_topic_policy: gentle | skip`.
- **Never repeats** a topic within `avoid_repeat_hours` (keyword overlap on `data/memory.json`).

## Training Peewee on your own images

Yes! See [`train/README.md`](train/README.md). Short version: 20–60 images → `prepare_dataset.py`
→ `train_lora.sh` on a rented GPU for ~30 min → point `painter.local.lora_path` at the result.
You can train a *style* LoRA (paint like your images) and/or a *character* LoRA (put Peewee in the scene).

## Repo map

```
peewee/          the bot            gallery/    the website (deployed as-is)
  main.py        CLI + scheduler      index.html  feed.json  paintings/  assets/
  trends.py      sources            train/      LoRA training kit
  brain.py       Claude planner     deploy/     Dockerfile, fly.toml, systemd, launchd
  painter.py     SD backends        data/       memory.json (topic history)
  publish_*.py   site + X
config.yaml      all behaviour      .env        all secrets (never committed)
```
