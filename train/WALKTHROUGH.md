# Training Peewee on your artwork — step by step

The dataset is built: `train/peewee-dataset.zip` holds 110 captioned 1024×1024 crops made from your 60
"Peewee's memories" pieces (wide paintings were cut into 2–3 crops). Every crop has a hand-written caption
starting with the trigger word `pwmemories`. The zip is the only thing you upload to the GPU box.

Training needs an NVIDIA GPU. Your Mac can't do this in reasonable time. Two routes:

---

## Route A — RunPod, ~1 hour, ~$2–3 (recommended: the LoRA plugs into your Mac's local painter)

1. **Push your project to GitHub first** (see README step 5) so the GPU box can clone it.
   `train/data/` is gitignored, so zip the dataset separately: right-click `train/data/ready` → Compress.

2. **runpod.io** → sign up, add $10 credit → **Pods** → **Deploy**.
   Pick an **RTX 4090** (24 GB) or **A40/L4** — anything ≥ 16 GB VRAM.
   Template: *RunPod PyTorch 2.x*. Container disk 40 GB. Deploy.

3. When it says Running → **Connect** → **Start Web Terminal** (a terminal in your browser).

4. In that terminal:
   ```bash
   git clone https://github.com/YOUR-USERNAME/peewee.git
   cd peewee
   pip install -r train/requirements-train.txt
   ```

5. Upload the dataset: Connect → **Jupyter Lab**, navigate into `peewee/train/data/`, drag `peewee-dataset.zip`
   into it, then back in the terminal:
   ```bash
   cd train/data && unzip peewee-dataset.zip && cd ../..
   ls train/data/ready | head      # should list 001.jpg 001.txt …
   ```

6. Train:
   ```bash
   accelerate config default
   bash train/train_lora.sh
   ```
   ~35–45 minutes on a 4090 for the 110-image set at 1500 steps. You'll see a step counter.
   If it runs out of memory, edit `train/train_lora.sh` and add `--resolution=768`.

7. Download the result: in Jupyter Lab open `peewee/train/output/peewee-memories/` and right-click
   `pytorch_lora_weights.safetensors` → Download (~50–100 MB).

8. **Stop the pod** (Pods → Stop, then Terminate) so you stop paying.

9. On your Mac, put the file at `Downloads/peewee/train/output/peewee-memories/pytorch_lora_weights.safetensors`
   and in `config.yaml`:
   ```yaml
   painter:
     local:
       lora_path: train/output/peewee-memories/pytorch_lora_weights.safetensors
       lora_scale: 0.8
       trigger_word: pwmemories
   ```
   `python -m peewee.main once --dry` — the log will say `LoRA loaded`.

Tuning: if paintings look like copies of the references, lower `lora_scale` to 0.6. If the style is
faint, raise it to 1.0 or retrain with `STEPS=2500 bash train/train_lora.sh`. With 110 images, 1500 steps is a sensible first run; 2200 if the style comes out faint.

---

## Route B — Replicate, no terminal on a GPU box, but Peewee then paints via Replicate

1. replicate.com → sign in → add billing.
2. Open the **ostris/flux-dev-lora-trainer** model → **Train**. Upload `peewee-dataset.zip`,
   set trigger word `pwmemories`, steps 1000, leave the rest. Cost ≈ $2–4, ~20 min.
3. It gives you a model/LoRA URL. In `.env` set `REPLICATE_API_TOKEN`, and in `config.yaml`:
   ```yaml
   painter:
     backend: replicate
     replicate:
       lora_url: <the URL it gave you>
   ```
Every painting then costs a few cents and Peewee no longer needs your Mac's GPU at all — that's
the trade: simpler, but a small recurring bill (48/day ≈ $1.50/day).
