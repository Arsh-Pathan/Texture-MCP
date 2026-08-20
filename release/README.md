---
license: creativeml-openrail-m
base_model: stable-diffusion-v1-5/stable-diffusion-v1-5
tags:
- text-to-image
- stable-diffusion
- diffusers
- minecraft
- pixel-art
- texture
- 16x16
library_name: diffusers
pipeline_tag: text-to-image
---

# Minecraft Item Textures 16px

A full fine-tune of the Stable Diffusion 1.5 UNet that produces **real 16×16 Minecraft mod
item textures** with transparency, from free-text prompts.

Unlike pixel-art style LoRAs, the output is not "an illustration that looks like pixel art".
It is an actual 16×16 RGBA sprite you can drop into a resource pack.

![samples](https://huggingface.co/yuuki14202028/minecraft-item-16px/resolve/main/preview.png)

## The one thing you must know

**Generate at 256×256, then fold each 16×16 block into one pixel.**

The model was trained exclusively on 16×16 textures upscaled ×16 with nearest-neighbour.
Every training image therefore consists of 16×16 flat blocks of 16×16 pixels each, so the
model only ever draws on that grid. Folding the 256px output by block-averaging recovers the
original 16×16 sprite with no loss.

Transparency is encoded as **magenta (255, 0, 255)** and keyed out afterwards. That colour was
chosen because only 0.057% of opaque pixels in the training data come near it (black would
collide with outlines in 23% of images, white in 25%).

**Use an empty negative prompt.** Training used caption dropout to an empty string, so the
empty string is the model's unconditional path. Non-empty negatives contain words that are
almost absent from the 11,613-word training vocabulary, and pushing away from them injects
noise. Measured: with the usual `"blurry, photo, watermark…"` negative, `usb drive` and `cat`
never resolve at any guidance scale; with an empty negative both work.

## Usage

```python
import numpy as np, torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

BG = np.array([255, 0, 255], dtype=np.int16)   # chroma key colour

pipe = StableDiffusionPipeline.from_pretrained(
    "yuuki14202028/minecraft-item-16px", torch_dtype=torch.float16,
    safety_checker=None, requires_safety_checker=False).to("cuda")   # or "mps"
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

image = pipe("copper sword", negative_prompt="", num_inference_steps=30,
             guidance_scale=9.0, height=256, width=256).images[0]

def to_tile(image, size=16, tol=100):
    a = np.asarray(image.convert("RGB"), dtype=np.float32)
    c = a.shape[0] // size
    rgb = a.reshape(size, c, size, c, 3).mean(axis=(1, 3)).round().clip(0, 255).astype(np.int16)
    alpha = np.where(np.abs(rgb - BG).sum(axis=2) <= tol, 0, 255).astype(np.uint8)
    return Image.fromarray(np.dstack([rgb.astype(np.uint8), alpha]), "RGBA")

to_tile(image).save("copper_sword.png")        # 16x16 RGBA
```

Recommended settings: **256×256, guidance 9.0, 30 steps, empty negative prompt, DPM-Solver++**.
Guidance 7–12 all work; 3–5 is visibly mushy.

## What it can and cannot do

Measured on 50 prompts × 2 seeds and a 20-prompt × 3-seed diagnostic set.

**Strong.** Material × form composition is essentially perfect, including pairs that never
co-occur in the training data: `glass hammer` (glass and hammer appear separately, never
together) and `copper crown` both hit 3/3. `diamond crown`, `emerald key`, `gold skull`,
`glass sword` — 16/16.

**Good.** Objects absent from Minecraft: coffee cup, donut, cheese, syringe, gear, umbrella,
sushi, penguin, jellyfish, torii gate, katana. Abstract prompts like `black hole` and
`rainbow` also produce sensible sprites. Multi-adjective prompts (`glowing molten iron sword`,
`ancient mossy stone tablet`) mostly land.

**Weak.** Concepts with zero training-vocabulary support rely entirely on the SD1.5 prior and
hit roughly 2 in 3 (`guitar`, `microscope`). Fine-grained modifiers such as "tiny" and "rusty"
are often ignored. Around 10–15% of generations fill the whole frame instead of leaving a
background, so the chroma key finds nothing — regenerate with another seed.

**Judge by hit rate, not by one image.** Seed variance is large. Single samples repeatedly
gave the wrong impression during development: `steel ingot` looked broken until 4 seeds showed
3/4 correct, and `banana` looked impossible until 6 seeds showed 5/6 correct.

## Training

| | |
|---|---|
| Base | `stable-diffusion-v1-5/stable-diffusion-v1-5`, UNet fully unfrozen (859.5M) |
| Frozen | VAE (83.7M), CLIP text encoder (123.1M) |
| Data | 369,342 item textures, 16×16 RGBA, upscaled ×16 to 256px |
| Captions | derived from file names (`copper_sword.png` → "copper sword"), 11,613-word vocabulary |
| Steps | 92,335 (4 epochs, batch 16), lr 1e-5 cosine, 500 warmup |
| Objective | epsilon prediction with min-SNR weighting (γ=5), 10% caption dropout |
| Hardware | 1× RTX 4090, 6.2 hours, 66.8 img/s |

Loss over the run: 0.01192 → 0.00858, flattening in the final quarter as the cosine schedule
decayed, which suggests 4 epochs was about the right length for this configuration.

## Training data provenance and licensing

Trained on [`OVAWARE/16xModdedMinecraft`](https://huggingface.co/datasets/OVAWARE/16xModdedMinecraft),
a gated dataset of 1,034,057 textures scraped from Minecraft mods on Modrinth. Only the
552,357 entries labelled `item` were used, reduced to 369,342 after removing exact duplicates,
near-transparent and single-colour images, UI elements, and entries whose file names yielded no
usable caption.

**The dataset carries 235 distinct license values, including `LicenseRef-All-Rights-Reserved`.**
The weights are not a copy of any texture, but generated outputs may resemble existing mod
assets. If you redistribute generated textures — in a resource pack, a mod, or a commercial
product — verify their originality yourself. This model is released for research and personal
use; the authors of the original textures did not consent to this specific use.

Base model weights are CreativeML OpenRAIL-M and that license carries over here.

## Limitations

- Items only. Full-bleed block textures were deliberately excluded from training.
- 16×16 only. There is no meaningful detail to extract at higher folds.
- English prompts only, and the caption vocabulary is skewed toward modded Minecraft terms.
- No EMA was used during training; adding it would likely improve sample quality.
- The text encoder was left frozen. Unfreezing it at a low learning rate is the most obvious
  next experiment for prompt adherence.

## Code

Training, sampling, evaluation and img2img tooling:
[github.com/yuuki14202028/pixelgen](https://github.com/yuuki14202028/pixelgen)

---

### 日本語

Stable Diffusion 1.5 のUNetを、Minecraft MODのアイテムテクスチャ369,342枚でフルファインチューンしたモデルです。
**出力は「ピクセルアート風の絵」ではなく、リソースパックにそのまま入る本物の16×16 RGBAテクスチャ**です。

使い方の要点は3つ。**256×256で生成し、16×16画素ずつ平均して畳む**こと。**透過はマゼンタ(255,0,255)**で表現されているのでキーイングして戻すこと。そして
**negative promptは空にする**こと（学習時の無条件経路が空文字列なので、非空のnegativeはノイズになります）。

推奨設定は 256×256 / guidance 9.0 / 30ステップ / negative空 / DPM-Solver++ です。
