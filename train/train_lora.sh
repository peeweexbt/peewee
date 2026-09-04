#!/usr/bin/env bash
# Train an SDXL LoRA on train/data/ready using diffusers' reference script.
# Requires an NVIDIA GPU (>=16GB VRAM). ~25 min on an RTX 4090 for 30 images / 1500 steps.
set -euo pipefail
cd "$(dirname "$0")/.."

NAME="${NAME:-peewee-style}"
DATA="${DATA:-train/data/ready}"
STEPS="${STEPS:-1500}"
RANK="${RANK:-16}"
MODEL="stabilityai/stable-diffusion-xl-base-1.0"
VAE="madebyollin/sdxl-vae-fp16-fix"

if [ ! -f train/train_dreambooth_lora_sdxl.py ]; then
  echo "fetching diffusers training script…"
  curl -sSL -o train/train_dreambooth_lora_sdxl.py \
    https://raw.githubusercontent.com/huggingface/diffusers/main/examples/dreambooth/train_dreambooth_lora_sdxl.py
fi

# diffusers' dreambooth script reads captions from a dataset dir when --caption_column is used via
# an imagefolder dataset; build the metadata.jsonl it expects.
python - <<'PY'
import json, pathlib, os
d = pathlib.Path(os.environ.get("DATA", "train/data/ready"))
with open(d / "metadata.jsonl", "w") as f:
    for p in sorted(d.glob("*.png")):
        cap = p.with_suffix(".txt").read_text().strip() if p.with_suffix(".txt").exists() else "pwstyle"
        f.write(json.dumps({"file_name": p.name, "caption": cap}) + "\n")
print("metadata.jsonl written")
PY

accelerate launch train/train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path="$MODEL" \
  --pretrained_vae_model_name_or_path="$VAE" \
  --dataset_name="$DATA" \
  --caption_column="caption" \
  --instance_prompt="pwstyle, an impressionist painting" \
  --output_dir="train/output/$NAME" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps="$STEPS" \
  --rank="$RANK" \
  --mixed_precision="fp16" \
  --use_8bit_adam \
  --checkpointing_steps=500 \
  --seed=42

echo
echo "done → train/output/$NAME/pytorch_lora_weights.safetensors"
echo "set painter.local.lora_path to that file in config.yaml"
