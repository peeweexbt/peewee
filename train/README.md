# Teaching Peewee your style (custom images → LoRA)

Yes — you can train Peewee on your own images. The standard way is a **LoRA**: a small
adapter file (~50–200 MB) that sits on top of Stable Diffusion XL and nudges it toward
your images. You don't retrain the whole model, so it takes 20–60 minutes on one GPU.

Two flavours, pick one (or do both and stack them):

| Goal | What to feed it | Trigger word |
|---|---|---|
| **Style** — "paint like *these*" (your own Monet-ish canvases, a specific palette, a brush feel) | 20–60 images that share the style; varied subjects | `pwstyle` |
| **Character** — put Peewee himself into paintings | 15–30 images of Peewee from different angles / poses | `pwcat` |

## 1. Collect images
- 1024×1024 is ideal; anything ≥ 768 px works. Square-ish crops train best.
- Quality > quantity. Drop blurry, watermarked or near-duplicate images.
- Put them in `train/data/raw/`.

## 2. Prepare + caption
```bash
pip install -r train/requirements-train.txt
python train/prepare_dataset.py --trigger pwstyle
```
This crops/resizes to 1024 and writes a caption `.txt` next to each image
(BLIP auto-captions, with your trigger word prepended). **Open a few captions and fix them** —
good captions = good LoRA. Describe the *content* in the caption; the trigger word carries the style.

## 3. Train
You need an NVIDIA GPU with ≥ 16 GB VRAM (24 GB comfortable). Options:
- **Rent one for an hour**: RunPod / Lambda / Vast.ai, pick an A10G/L4/RTX 4090 pod, clone this repo, run the script. Cost is a couple of dollars.
- **Your Mac**: Apple Silicon *can* train SDXL LoRAs with diffusers on MPS but it is very slow (hours) and memory-hungry. Not recommended for the first attempt.
- **Replicate**: upload a zip of your images to `ostris/flux-dev-lora-trainer` (or an SDXL trainer) and it gives you a LoRA URL — no GPU, ~$2–4. Then set `painter.backend: replicate` and `replicate.lora_url`.

```bash
bash train/train_lora.sh            # ~25 min on a 4090 with 30 images
```
Output: `train/output/peewee-style/pytorch_lora_weights.safetensors`

## 4. Use it
In `config.yaml`:
```yaml
painter:
  local:
    lora_path: train/output/peewee-style/pytorch_lora_weights.safetensors
    lora_scale: 0.8          # 0.6 subtle … 1.0 strong
    trigger_word: pwstyle
```
Run `python -m peewee.main once --dry` and compare against `lora_scale: 0` to see the effect.

## Tips
- If results look *too* much like your training images (copies), lower `lora_scale` or train fewer steps.
- If the style barely shows, raise `--max_train_steps` (1500 → 2500) or `lora_scale`.
- Keep Monet in the style prompt — the LoRA and the text prompt add together.
- Stacking: train `pwstyle` and `pwcat` separately, then load both with `pipe.load_lora_weights` twice
  and `pipe.set_adapters(["style","cat"], [0.8, 0.7])` — see diffusers docs for multi-LoRA.

## Legal note
Train only on images you own or have permission to use. Monet's own works are public domain,
so paintings of his you photograph/download from museum open-access collections (Met, Art Institute of Chicago,
Rijksmuseum) are fair game for a style set.
