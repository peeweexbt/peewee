"""Crop/resize your images to 1024² and auto-caption them for LoRA training.

python train/prepare_dataset.py --trigger pwstyle [--src train/data/raw] [--dst train/data/ready] [--no-caption]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".tif", ".tiff", ".bmp"}


def center_square(img: Image.Image, size: int) -> Image.Image:
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    return img.resize((size, size), Image.LANCZOS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="train/data/raw")
    ap.add_argument("--dst", default="train/data/ready")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--trigger", default="pwstyle")
    ap.add_argument("--no-caption", action="store_true", help="skip BLIP; write only the trigger word")
    a = ap.parse_args()

    src, dst = Path(a.src), Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in EXTS)
    if not files:
        raise SystemExit(f"no images in {src}")

    captioner = None
    if not a.no_caption:
        try:
            import torch
            from transformers import BlipForConditionalGeneration, BlipProcessor

            dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
            proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(dev)

            def captioner(img):  # noqa: E306
                inputs = proc(img, "an impressionist painting of", return_tensors="pt").to(dev)
                out = model.generate(**inputs, max_new_tokens=40)
                return proc.decode(out[0], skip_special_tokens=True)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] BLIP unavailable ({e}); writing trigger-only captions")

    for i, p in enumerate(files, 1):
        img = center_square(Image.open(p), a.size)
        out = dst / f"{i:03d}.png"
        img.save(out)
        cap = a.trigger
        if captioner:
            try:
                cap = f"{a.trigger}, {captioner(img)}"
            except Exception as e:  # noqa: BLE001
                print(f"[warn] caption failed for {p.name}: {e}")
        out.with_suffix(".txt").write_text(cap)
        print(f"{p.name} → {out.name}: {cap}")
    print(f"\n{len(files)} images ready in {dst}. Review the .txt captions before training!")


if __name__ == "__main__":
    main()
