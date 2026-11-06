"""Modern Web UI for Texture-Craft Minecraft 16x16 Item Texture Generator.
Runs on RTX 4050 GPU using the local fine-tuned model.
"""

import os
import sys
import random
import tempfile
import zipfile
import pathlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import gradio as gr
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    DPMSolverMultistepScheduler,
)

sys.path.insert(0, ".")
import mcpixel

LOCAL_MODEL_DIR = pathlib.Path("models/minecraft-item-16px")
MAGIC_ITEMS_DIR = pathlib.Path("magic_items/tile")
BG = np.array([255, 0, 255], dtype=np.int16)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"[WebUI] Loading pipeline on {device} from {LOCAL_MODEL_DIR}...")
pipe = StableDiffusionPipeline.from_pretrained(
    str(LOCAL_MODEL_DIR),
    torch_dtype=dtype,
    safety_checker=None,
    requires_safety_checker=False,
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)
pipe.set_progress_bar_config(disable=True)

# Share components for img2img to avoid extra VRAM usage
i2i_pipe = StableDiffusionImg2ImgPipeline(**pipe.components)
i2i_pipe.set_progress_bar_config(disable=True)
print("[WebUI] Ready!")


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


def create_checkered_preview(tile: Image.Image, cell_size: int = 16, target_size: int = 256) -> Image.Image:
    """Render crisp 16x16 tile on checkerboard background for display."""
    upscaled = tile.resize((target_size, target_size), Image.Resampling.NEAREST)
    canvas = Image.new("RGBA", (target_size, target_size), (40, 40, 48, 255))
    
    # Checkered pattern
    for y in range(0, target_size, cell_size):
        for x in range(0, target_size, cell_size):
            if (x // cell_size + y // cell_size) % 2:
                canvas.paste((65, 65, 75, 255), (x, y, x + cell_size, y + cell_size))
                
    canvas.alpha_composite(upscaled)
    return canvas.convert("RGB")


def to_init(image: Image.Image, size: int = 16, up: int = 16) -> Image.Image:
    """Format sketch canvas to match training format for img2img."""
    im = image.convert("RGBA")
    w, h = im.size
    if w % size or h % size:
        im = im.resize(
            (max(w // size, 1) * size, max(h // size, 1) * size),
            Image.Resampling.NEAREST,
        )
    a = np.asarray(im, dtype=np.float32)
    bh, bw = a.shape[0] // size, a.shape[1] // size
    a = a.reshape(size, bh, size, bw, 4)
    solid = a[..., 3] >= 128
    cnt = solid.sum(axis=(1, 3))
    rgb = (a[..., :3] * solid[..., None]).sum(axis=(1, 3)) / np.maximum(cnt, 1)[..., None]
    tile = np.where((cnt > 0)[..., None], rgb, BG.astype(np.float32))
    tile = tile.round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(tile).resize((size * up, size * up), Image.Resampling.NEAREST)


def generate_textures(prompt, num_variants, guidance, steps, seed):
    if not prompt or not prompt.strip():
        return [], None, "Please enter a prompt."

    prompt = prompt.strip()
    if seed is None or seed == -1:
        seed = random.randint(0, 999999)

    out_temp = Path(tempfile.mkdtemp())
    safe_name = "".join(c if c.isalnum() else "_" for c in prompt)[:30] or "item"

    previews = []
    file_paths = []

    for k in range(int(num_variants)):
        current_seed = int(seed) + k * 1000
        gen = torch.Generator(device="cpu").manual_seed(current_seed)
        im = pipe(
            prompt,
            negative_prompt="",
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            height=256,
            width=256,
            generator=gen,
        ).images[0]

        # 16x16 tile with transparency
        tile = key_out(fold_to_tile(im, 16))
        tile_path = out_temp / f"{safe_name}_v{k+1}.png"
        tile.save(tile_path)
        file_paths.append(str(tile_path))

        # Checkered crisp preview
        preview = create_checkered_preview(tile)
        previews.append(preview)

    # Create zip file for one-click bulk download
    zip_path = out_temp / f"{safe_name}_textures.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for fp in file_paths:
            zf.write(fp, arcname=Path(fp).name)

    info = f"Generated {num_variants} texture(s) for '{prompt}' (Seed: {seed})"
    return previews, str(zip_path), info


def generate_from_sketch(sketch, prompt, strength, guidance, steps, seed):
    if sketch is None:
        return None, None, "Please draw a sketch first!"
    if not prompt or not prompt.strip():
        prompt = "minecraft item"

    if isinstance(sketch, dict):
        img = sketch.get("composite") or sketch.get("background")
    else:
        img = sketch

    if img is None:
        return None, None, "No sketch image found."

    if seed is None or seed == -1:
        seed = random.randint(0, 999999)

    init_img = to_init(img)
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    
    im = i2i_pipe(
        prompt=prompt.strip(),
        image=init_img,
        strength=float(strength),
        guidance_scale=float(guidance),
        num_inference_steps=int(steps),
        negative_prompt="",
        generator=gen,
    ).images[0]

    tile = key_out(fold_to_tile(im, 16))
    out_temp = Path(tempfile.mkdtemp())
    tile_path = out_temp / "sketch_textured.png"
    tile.save(tile_path)
    preview = create_checkered_preview(tile)
    return preview, str(tile_path), f"Texture rendered from sketch with seed {seed}!"


def load_magic_library():
    """Load pre-generated custom plugin items for instant preview."""
    if not MAGIC_ITEMS_DIR.exists():
        return []
    items = []
    for p in sorted(MAGIC_ITEMS_DIR.glob("*.png")):
        try:
            im = Image.open(p)
            preview = create_checkered_preview(im)
            items.append((preview, p.stem.replace("_", " ").title()))
        except Exception:
            continue
    return items


# Build Web Interface
custom_css = """
body { background-color: #121316; }
.gradio-container { max-width: 1100px !important; margin: auto; }
.header-box { text-align: center; margin-bottom: 20px; }
.header-box h1 { font-size: 2.2rem; font-weight: 700; color: #58a6ff; }
.header-box p { color: #8b949e; font-size: 1rem; }
.gallery-preview { border-radius: 8px; overflow: hidden; }
"""

with gr.Blocks(title="Texture-Craft | Minecraft Item Generator") as demo:
    gr.HTML(
        """
        <div class="header-box">
            <h1>✨ Texture-Craft Studio</h1>
            <p>Generate native 16×16 transparent pixel-art textures for Minecraft mods and plugins in ~1.5s using your RTX 4050</p>
        </div>
        """
    )

    with gr.Tabs():
        # TAB 1: Text to Texture
        with gr.Tab("🎨 Text to Texture"):
            with gr.Row():
                with gr.Column(scale=4):
                    prompt_input = gr.Textbox(
                        label="Item Prompt",
                        placeholder="e.g. enchanted ruby sword, lightning wand, void crystal, mana potion...",
                        lines=2,
                    )
                    with gr.Row():
                        variants_slider = gr.Slider(
                            minimum=1, maximum=6, value=2, step=1, label="Variants"
                        )
                        seed_input = gr.Number(
                            value=-1, label="Seed (-1 for random)", precision=0
                        )

                    with gr.Accordion("Advanced Generation Settings", open=False):
                        with gr.Row():
                            guidance_slider = gr.Slider(
                                minimum=3.0, maximum=15.0, value=9.0, step=0.5, label="Guidance Scale"
                            )
                            steps_slider = gr.Slider(
                                minimum=15, maximum=50, value=30, step=5, label="Steps"
                            )

                    generate_btn = gr.Button("🚀 Generate Textures", variant="primary", size="lg")

                    gr.Examples(
                        examples=[
                            ["flame sword", 2, 9.0, 30],
                            ["enchanted diamond dagger", 2, 9.0, 30],
                            ["blue mana potion", 2, 9.0, 30],
                            ["glowing purple crystal", 2, 9.0, 30],
                            ["golden wizard wand", 2, 9.0, 30],
                            ["void relic amulet", 2, 9.0, 30],
                            ["elder staff", 2, 9.0, 30],
                            ["celestial spell book", 2, 9.0, 30],
                        ],
                        inputs=[prompt_input, variants_slider, guidance_slider, steps_slider],
                        label="Quick Item Presets",
                    )

                with gr.Column(scale=5):
                    status_output = gr.Markdown("Ready to generate.")
                    gallery_output = gr.Gallery(
                        label="Crisp 16x16 Pixel Art Preview (with transparency)",
                        columns=3,
                        rows=2,
                        height=360,
                        preview=True,
                    )
                    download_output = gr.File(label="📦 Download 16x16 PNGs (ZIP)")

            generate_btn.click(
                fn=generate_textures,
                inputs=[prompt_input, variants_slider, guidance_slider, steps_slider, seed_input],
                outputs=[gallery_output, download_output, status_output],
            )

        # TAB 2: Sketch to Texture
        with gr.Tab("✏️ Sketch to Texture"):
            gr.Markdown("Draw a rough 16x16 shape or outline, and let the model add shading, materials, and depth!")
            with gr.Row():
                with gr.Column():
                    sketch_canvas = gr.ImageEditor(
                        label="Doodle Pad (Draw Item Silhouette)",
                        type="pil",
                        image_mode="RGBA",
                        brush=gr.Brush(colors=["#ffffff", "#e74c3c", "#3498db", "#2ecc71", "#f1c40f", "#9b59b6", "#111111"], default_size=8),
                    )
                    sketch_prompt = gr.Textbox(
                        label="What should this be?",
                        value="crystal sword",
                        placeholder="e.g. fire battleaxe, wooden wand, healing flask",
                    )
                    sketch_strength = gr.Slider(
                        minimum=0.4, maximum=0.95, value=0.75, step=0.05, label="Transformation Strength"
                    )
                    sketch_seed = gr.Number(value=-1, label="Seed (-1 for random)", precision=0)
                    sketch_btn = gr.Button("✨ Render Texture from Sketch", variant="primary")

                with gr.Column():
                    sketch_status = gr.Markdown("Waiting for sketch...")
                    sketch_preview = gr.Image(label="Finished 16x16 Texture (Pixel Preview)")
                    sketch_file = gr.File(label="Download 16x16 PNG")

            sketch_btn.click(
                fn=generate_from_sketch,
                inputs=[sketch_canvas, sketch_prompt, sketch_strength, guidance_slider, steps_slider, sketch_seed],
                outputs=[sketch_preview, sketch_file, sketch_status],
            )

        # TAB 3: Plugin Magic Library
        with gr.Tab("💎 Pre-generated Plugin Items"):
            gr.Markdown("Here are the custom magical items generated earlier for your Minecraft plugin! All ready in `magic_items/tile/`:")
            library_gallery = gr.Gallery(
                value=load_magic_library(),
                label="Library Items (40 textures)",
                columns=5,
                height=450,
            )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
