"""Generate 16x16 magical item textures using the minecraft-item-16px model.
Saves model locally to ./models/minecraft-item-16px and uses authenticated HF token.
"""

import os
import sys
import pathlib
import time
import numpy as np
import torch
from PIL import Image

# Use HF Token from environment if available
HF_TOKEN = os.environ.get("HF_TOKEN", None)

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

sys.path.insert(0, ".")
import mcpixel

BG = np.array([255, 0, 255], dtype=np.int16)
NEGATIVE = ""
LOCAL_MODEL_DIR = pathlib.Path("models/minecraft-item-16px")
REPO_ID = "yuuki14202028/minecraft-item-16px"


def fold_to_tile(image: Image.Image, size: int = 16) -> Image.Image:
    """Fold 256px output back to size x size pixel art."""
    a = np.asarray(image.convert("RGB"), dtype=np.float32)
    c = a.shape[0] // size
    return Image.fromarray(
        a.reshape(size, c, size, c, 3)
        .mean(axis=(1, 3))
        .round()
        .clip(0, 255)
        .astype(np.uint8)
    )


def key_out(tile: Image.Image, tol: int = 100) -> Image.Image:
    """Chroma-key magenta background into clean transparency."""
    rgb = np.asarray(tile.convert("RGB"), dtype=np.int16)
    bg = np.abs(rgb - BG).sum(axis=2) <= tol
    out = np.dstack([rgb.astype(np.uint8), np.where(bg, 0, 255).astype(np.uint8)])
    return Image.fromarray(out)


def main() -> None:
    # Custom items for the Minecraft magic plugin
    prompts = [
        # Swords
        "enchanted diamond sword",
        "crystal sword",
        "flame sword",
        "frost sword",
        "void sword",
        # Potions
        "blue mana potion",
        "red health potion",
        "green potion",
        "purple mystic potion",
        # Crystals & Gems
        "glowing crystal",
        "amethyst crystal",
        "fire crystal",
        "dark void crystal",
        # Wands & Staffs
        "magic wand",
        "crystal wand",
        "golden wand",
        "wizard staff",
        # Relics & Accessories
        "enchanted book",
        "magic orb",
        "glowing amulet",
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"[info] Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})", flush=True)

    if LOCAL_MODEL_DIR.exists() and any(LOCAL_MODEL_DIR.iterdir()):
        print(f"[info] Loading saved model from local directory: {LOCAL_MODEL_DIR}", flush=True)
        pipe = StableDiffusionPipeline.from_pretrained(
            str(LOCAL_MODEL_DIR),
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
    else:
        print(f"[info] Downloading model from Hugging Face ({REPO_ID}) with authenticated token...", flush=True)
        pipe = StableDiffusionPipeline.from_pretrained(
            REPO_ID,
            token=HF_TOKEN,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        print(f"[info] Saving model locally to {LOCAL_MODEL_DIR}...", flush=True)
        LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        pipe.save_pretrained(str(LOCAL_MODEL_DIR))
        print(f"[info] Model saved successfully to {LOCAL_MODEL_DIR}!", flush=True)

    pipe = pipe.to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    outdir = pathlib.Path("magic_items")
    for sub in ("raw", "tile"):
        (outdir / sub).mkdir(parents=True, exist_ok=True)

    tiles, labels = [], []
    seed_base = 42
    n_per_prompt = 2
    total_images = len(prompts) * n_per_prompt

    print(f"\n[gen] Generating {total_images} custom textures ({len(prompts)} items x {n_per_prompt} variants)...", flush=True)
    count = 0
    start_time = time.time()

    for i, prompt in enumerate(prompts):
        for k in range(n_per_prompt):
            count += 1
            t0 = time.time()
            gen = torch.Generator(device="cpu").manual_seed(seed_base + i * 1000 + k)
            im = pipe(
                prompt,
                negative_prompt=NEGATIVE,
                num_inference_steps=30,
                guidance_scale=9.0,
                height=256,
                width=256,
                generator=gen,
            ).images[0]

            name = prompt.replace(" ", "_") + (f"_var{k+1}" if n_per_prompt > 1 else "")
            im.save(outdir / "raw" / f"{name}.png")
            tile = key_out(fold_to_tile(im, 16))
            tile.save(outdir / "tile" / f"{name}.png")
            tiles.append(tile)
            labels.append(name)
            dt = time.time() - t0
            print(f"[{count:02d}/{total_images:02d}] Generated '{name}' in {dt:.2f}s", flush=True)

    total_time = time.time() - start_time
    print(f"\n[info] All {total_images} textures generated in {total_time:.1f}s!", flush=True)

    # Generate contact sheet
    cols = 5
    contact_path = outdir / "contact_sheet.png"
    mcpixel.contact_sheet(tiles, cols=cols, cell=96).save(contact_path)
    outdir.joinpath("index.txt").write_text(
        "\n".join(f"Row {i//cols+1}, Col {i%cols+1}: {n}" for i, n in enumerate(labels)),
        encoding="utf8",
    )
    print(f"[done] Contact sheet saved to {contact_path}", flush=True)
    print(f"[done] 16x16 PNG textures saved to {outdir / 'tile'}", flush=True)


if __name__ == "__main__":
    main()
