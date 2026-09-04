"""Turn a Plan into pixels.

Backends (config.painter.backend):
  local      – Stable Diffusion via diffusers, in-process (CUDA / Apple MPS / CPU). Supports your LoRA.
  remote_sd  – POST to an AUTOMATIC1111/Forge (--api) or ComfyUI server you run somewhere with a GPU.
  replicate  – hosted fallback.
  mock       – a procedural, Monet-ish "painting" so the whole pipeline can be tested with no GPU.
"""
from __future__ import annotations

import base64
import io
import logging
import math
import random
import time
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFilter

from .config import env

log = logging.getLogger("peewee.painter")


def full_prompt(cfg, subject: str, colour_words: list[str] | None = None) -> str:
    style = " ".join(cfg.painter.style_prompt.split())
    if colour_words:
        subject = f"{subject.strip().rstrip('.')}, {', '.join(colour_words[:3])}"
    trig = cfg.painter.local.trigger_word if cfg.painter.backend == "local" and cfg.painter.local.lora_path else ""
    # style FIRST — CLIP truncates at 77 tokens, so the subject is what gets clipped if anything does
    words = subject.strip().rstrip(".").split()
    subject = " ".join(words[:42])
    parts = [p for p in (trig, style, subject) if p]
    return ", ".join(parts)


def negative(cfg) -> str:
    return " ".join(cfg.painter.negative_prompt.split())


def _seed(cfg) -> int:
    s = int(cfg.painter.seed)
    return random.randint(0, 2**31 - 1) if s < 0 else s


# ── local: diffusers ──────────────────────────────────────────────────────
_PIPE = None


def _load_local(cfg):
    global _PIPE
    if _PIPE is not None:
        return _PIPE
    import torch
    from diffusers import AutoPipelineForText2Image

    lc = cfg.painter.local
    dev = lc.device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dt = lc.dtype
    if dt == "auto":
        dt = "float16" if dev == "cuda" else ("float16" if dev == "mps" else "float32")
    dtype = getattr(torch, dt)
    log.info("loading %s on %s (%s)…", lc.model, dev, dt)
    pipe = AutoPipelineForText2Image.from_pretrained(lc.model, torch_dtype=dtype, use_safetensors=True)
    if lc.lora_path:
        pipe.load_lora_weights(lc.lora_path)
        pipe.fuse_lora(lora_scale=float(lc.lora_scale))
        log.info("LoRA loaded: %s (scale %.2f)", lc.lora_path, lc.lora_scale)
    pipe.to(dev)
    if dev == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:  # noqa: BLE001
            pass
    if dev in ("mps", "cpu"):
        pipe.enable_attention_slicing()
    _PIPE = pipe
    return pipe


def paint_local(cfg, prompt: str) -> Image.Image:
    import torch

    pipe = _load_local(cfg)
    seed = _seed(cfg)
    g = torch.Generator(device="cpu").manual_seed(seed)
    img = pipe(
        prompt=prompt,
        negative_prompt=negative(cfg),
        width=int(cfg.painter.width),
        height=int(cfg.painter.height),
        num_inference_steps=int(cfg.painter.steps),
        guidance_scale=float(cfg.painter.guidance),
        generator=g,
    ).images[0]
    log.info("painted locally (seed %d)", seed)
    return img


# ── remote_sd: A1111 / Forge / ComfyUI ────────────────────────────────────
def paint_remote(cfg, prompt: str) -> Image.Image:
    rc = cfg.painter.remote_sd
    if rc.kind == "a1111":
        payload = {
            "prompt": prompt + (" " + rc.lora_tag if rc.lora_tag else ""),
            "negative_prompt": negative(cfg),
            "width": int(cfg.painter.width),
            "height": int(cfg.painter.height),
            "steps": int(cfg.painter.steps),
            "cfg_scale": float(cfg.painter.guidance),
            "seed": _seed(cfg),
            "sampler_name": "DPM++ 2M Karras",
        }
        r = requests.post(f"{rc.url.rstrip('/')}/sdapi/v1/txt2img", json=payload, timeout=600)
        r.raise_for_status()
        return Image.open(io.BytesIO(base64.b64decode(r.json()["images"][0]))).convert("RGB")
    if rc.kind == "comfy":
        return _paint_comfy(cfg, prompt)
    raise ValueError(f"unknown remote_sd.kind {rc.kind}")


def _paint_comfy(cfg, prompt: str) -> Image.Image:
    """Minimal ComfyUI default-workflow call. Edit the graph below to match your checkpoint/LoRA."""
    rc = cfg.painter.remote_sd
    base = rc.url.rstrip("/")
    graph = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": negative(cfg), "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": int(cfg.painter.width), "height": int(cfg.painter.height), "batch_size": 1}},
        "5": {"class_type": "KSampler", "inputs": {"seed": _seed(cfg), "steps": int(cfg.painter.steps), "cfg": float(cfg.painter.guidance),
                                                  "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0,
                                                  "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0]}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"filename_prefix": "peewee", "images": ["6", 0]}},
    }
    pid = requests.post(f"{base}/prompt", json={"prompt": graph}, timeout=30).json()["prompt_id"]
    for _ in range(600):
        time.sleep(1)
        hist = requests.get(f"{base}/history/{pid}", timeout=30).json()
        if pid in hist:
            out = hist[pid]["outputs"]["7"]["images"][0]
            r = requests.get(f"{base}/view", params={"filename": out["filename"], "subfolder": out.get("subfolder", ""), "type": out.get("type", "output")}, timeout=60)
            return Image.open(io.BytesIO(r.content)).convert("RGB")
    raise TimeoutError("ComfyUI did not finish in 10 minutes")


# ── replicate ─────────────────────────────────────────────────────────────
def paint_replicate(cfg, prompt: str) -> Image.Image:
    import replicate

    rc = cfg.painter.replicate
    inp = {"prompt": prompt, "width": int(cfg.painter.width), "height": int(cfg.painter.height),
           "guidance": float(cfg.painter.guidance), "num_inference_steps": int(cfg.painter.steps), "output_format": "png"}
    if rc.lora_url:
        inp["hf_lora"] = rc.lora_url
    out = replicate.run(rc.model, input=inp)
    url = out[0] if isinstance(out, list) else out
    url = str(getattr(url, "url", url))
    return Image.open(io.BytesIO(requests.get(url, timeout=120).content)).convert("RGB")


# ── mock: procedural impressionism ────────────────────────────────────────
def paint_mock(cfg, prompt: str, palette: list[str] | None = None) -> Image.Image:
    """Dabs of colour with a soft horizon — good enough to see the pipeline & site work."""
    w, h = int(cfg.painter.width), int(cfg.painter.height)
    rnd = random.Random(_seed(cfg))
    cols = []
    for p in (palette or ["#8fb3d9", "#e9c9a8", "#b7a7d6", "#9fc7a3", "#f2e6c9"]):
        p = p.lstrip("#")
        if len(p) == 6:
            cols.append(tuple(int(p[i:i + 2], 16) for i in (0, 2, 4)))
    if not cols:
        cols = [(143, 179, 217), (233, 201, 168), (183, 167, 214)]
    img = Image.new("RGB", (w, h), cols[0])
    d = ImageDraw.Draw(img)
    horizon = int(h * rnd.uniform(0.35, 0.65))
    # sky/water gradient bands
    for y in range(h):
        t = y / h
        a, b = (cols[0], cols[1]) if y < horizon else (cols[1 % len(cols)], cols[2 % len(cols)])
        c = tuple(int(a[i] * (1 - t) + b[i] * t) for i in range(3))
        d.line([(0, y), (w, y)], fill=c)
    # thousands of brush dabs
    for _ in range(9000):
        c = rnd.choice(cols)
        jitter = tuple(max(0, min(255, v + rnd.randint(-28, 28))) for v in c)
        x, y = rnd.randint(0, w), rnd.randint(0, h)
        lw = rnd.randint(6, 26)
        lh = rnd.randint(2, 8)
        ang = rnd.uniform(-0.5, 0.5) if y < horizon else rnd.uniform(-0.1, 0.1)
        dx, dy = math.cos(ang) * lw, math.sin(ang) * lw
        d.line([(x, y), (x + dx, y + dy)], fill=jitter, width=lh)
    img = img.filter(ImageFilter.GaussianBlur(0.6))
    return img


# ── finishing pass: confetti dabs over the painting ───────────────────────
def dab_pass(img: Image.Image, strength: float = 0.7, density: float = 1.0, seed: int | None = None) -> Image.Image:
    """Confetti-dab finish: soften the painting into colour fields, then lay thousands of short
    rectangular dabs over it, coloured from the pixels beneath with lively jitter and pastel sparks.
    strength 0 = untouched, 1 = full mosaic. density scales the number of dabs.
    """
    rnd = random.Random(seed)
    base = img.convert("RGB")
    w, h = base.size
    scale = w / 1024
    soft = base.filter(ImageFilter.GaussianBlur(18 * scale))          # smooth colour fields under the dabs
    ground = Image.blend(base, soft, 0.85)
    sample = base.resize((max(1, w // 6), max(1, h // 6)), Image.BILINEAR)
    sw, sh = sample.size
    layer = ground.copy()
    d = ImageDraw.Draw(layer)
    count = int(10000 * density * (w * h) / (1024 * 1024))
    for _ in range(count):
        x, y = rnd.randint(0, w - 1), rnd.randint(0, h - 1)
        c = sample.getpixel((min(sw - 1, x // 6), min(sh - 1, y // 6)))
        r = rnd.random()
        if r < 0.18:      # pastel spark
            c = tuple(int(v * 0.45 + 255 * 0.55) for v in c)
        elif r < 0.30:    # deep accent
            c = tuple(int(v * 0.55) for v in c)
        c = tuple(max(0, min(255, v + rnd.randint(-38, 38))) for v in c)
        lw = rnd.randint(7, 28) * scale
        lh = max(1, int(rnd.randint(2, 9) * scale))
        ang = rnd.uniform(-0.55, 0.55) if y < h * 0.5 else rnd.uniform(-0.12, 0.12)
        d.line([(x, y), (x + math.cos(ang) * lw, y + math.sin(ang) * lw)], fill=c, width=lh)
    return Image.blend(base, layer, max(0.0, min(1.0, strength)))


# ── entry point ───────────────────────────────────────────────────────────
def paint(cfg, plan, out_path: Path) -> Path:
    prompt = full_prompt(cfg, plan.image_prompt, getattr(plan, "colour_words", None))
    backend = cfg.painter.backend
    log.info("painting with %s: %s", backend, prompt[:120])
    if backend == "local":
        img = paint_local(cfg, prompt)
    elif backend == "remote_sd":
        img = paint_remote(cfg, prompt)
    elif backend == "replicate":
        img = paint_replicate(cfg, prompt)
    elif backend == "mock":
        img = paint_mock(cfg, prompt, getattr(plan, "palette", None))
    else:
        raise ValueError(f"unknown painter backend {backend}")
    fin = cfg.painter.get("finish", {}) or {}
    if fin.get("dabs_chance", 0) > 0 and random.random() < float(fin["dabs_chance"]):
        strength = float(fin.get("dabs_strength", 0.55))
        img = dab_pass(img, strength=strength, density=float(fin.get("dabs_density", 1.0)))
        log.info("finish: dab pass (strength %.2f)", strength)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(out_path, "JPEG", quality=90, optimize=True)
    # web-sized thumb
    thumb = img.copy()
    thumb.thumbnail((640, 640))
    thumb.convert("RGB").save(out_path.with_name(out_path.stem + "_thumb.jpg"), "JPEG", quality=84, optimize=True)
    return out_path
