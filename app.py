"""Hugging Face Gradio Space and MCP tools for texture-only generation."""

from __future__ import annotations

import base64
import io
import os
import tempfile
import unicodedata
from pathlib import Path
from typing import Any

import gradio as gr
import spaces
from PIL import Image, ImageDraw

from texture_craft.engine import TextureEngine

MAX_PROMPT_LENGTH = 300
MAX_VARIANTS = 8
MAX_SEED = 2**31 - 1
MAX_STEPS = 50
IS_HF_SPACE = bool(os.getenv("SPACE_ID"))


def validate_generation_inputs(
    prompt: str,
    variants: int,
    seed: int,
    steps: int,
    guidance_scale: float,
) -> tuple[str, int, int, int, float]:
    """Validate and normalize public generation inputs."""
    if not isinstance(prompt, str):
        raise ValueError("Prompt must be text.")
    prompt = prompt.strip()
    if not prompt or len(prompt) > MAX_PROMPT_LENGTH:
        raise ValueError(f"Prompt must contain 1 to {MAX_PROMPT_LENGTH} characters.")
    if any(unicodedata.category(char).startswith("C") for char in prompt):
        raise ValueError("Prompt cannot contain control characters.")
    if not isinstance(variants, int) or isinstance(variants, bool) or not 1 <= variants <= MAX_VARIANTS:
        raise ValueError(f"Variants must be an integer from 1 to {MAX_VARIANTS}.")
    if not isinstance(seed, int) or isinstance(seed, bool) or not -1 <= seed <= MAX_SEED:
        raise ValueError(f"Seed must be -1 or an integer from 0 to {MAX_SEED}.")
    if not isinstance(steps, int) or isinstance(steps, bool) or not 10 <= steps <= MAX_STEPS:
        raise ValueError(f"Steps must be an integer from 10 to {MAX_STEPS}.")
    if not isinstance(guidance_scale, (int, float)) or isinstance(guidance_scale, bool):
        raise ValueError("Guidance scale must be a number from 3 to 15.")
    guidance_scale = float(guidance_scale)
    if not 3 <= guidance_scale <= 15:
        raise ValueError("Guidance scale must be a number from 3 to 15.")
    return prompt, variants, seed, steps, guidance_scale


def estimate_gpu_duration(
    prompt: str,
    variants: int,
    seed: int,
    steps: int,
    guidance_scale: float,
) -> int:
    """Estimate ZeroGPU allocation duration for bounded generation requests."""
    return min(300, max(60, 20 + variants * steps))


@spaces.GPU(duration=estimate_gpu_duration)
def _generate_on_gpu(
    prompt: str,
    variants: int,
    seed: int,
    steps: int,
    guidance_scale: float,
) -> list[dict[str, Any]]:
    engine = TextureEngine.get_instance(
        cache_downloaded_model=not IS_HF_SPACE,
        device="cuda" if IS_HF_SPACE else None,
    )
    return engine.generate_batch(
        prompt=prompt,
        variants=variants,
        seed_base=seed if seed >= 0 else None,
        steps=steps,
        guidance_scale=guidance_scale,
    )


def generate_textures(
    prompt: str,
    variants: int = 1,
    seed: int = -1,
    steps: int = 30,
    guidance_scale: float = 9.0,
) -> dict[str, Any]:
    """Generate transparent 16x16 item PNGs.

    Args:
        prompt: Short description of one item, up to 300 characters.
        variants: Number of images to generate, from 1 to 8.
        seed: Reproducible seed, or -1 to choose a random seed.
        steps: Diffusion steps, from 10 to 50.
        guidance_scale: Prompt guidance, from 3 to 15.

    Returns:
        Image filenames, seeds, timings, and base64 PNG/data-URI content.
    """
    prompt, variants, seed, steps, guidance_scale = validate_generation_inputs(
        prompt, variants, seed, steps, guidance_scale
    )
    results = _generate_on_gpu(prompt, variants, seed, steps, guidance_scale)

    textures: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="texture-craft-") as output_dir:
        for result in results:
            image = result["tile_image"].convert("RGBA")
            filename = f"texture-{result['variant_index']:02d}.png"
            image_path = Path(output_dir) / filename
            image.save(image_path, format="PNG")
            image_base64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
            textures.append(
                {
                    "filename": filename,
                    "variant_index": result["variant_index"],
                    "seed": result["seed"],
                    "latency_sec": result["latency_sec"],
                    "mime_type": "image/png",
                    "base64_png": image_base64,
                    "data_uri": f"data:image/png;base64,{image_base64}",
                }
            )

    return {"prompt": prompt, "count": len(textures), "textures": textures}


def _make_preview_sheet(textures: list[dict[str, Any]]) -> Image.Image:
    tile_size = 56
    gap = 8
    outer_margin = 8
    caption_height = 16
    columns = len(textures)
    sheet_width = outer_margin * 2 + columns * tile_size + (columns - 1) * gap
    sheet_height = outer_margin * 2 + tile_size + 4 + caption_height
    sheet = Image.new("RGBA", (sheet_width, sheet_height), (36, 37, 41, 255))
    draw = ImageDraw.Draw(sheet)

    for index, texture in enumerate(textures):
        x = outer_margin + index * (tile_size + gap)
        y = outer_margin
        draw.rounded_rectangle(
            (x - 3, y - 3, x + tile_size + 3, y + tile_size + 3),
            radius=4,
            fill=(28, 29, 32, 255),
            outline=(72, 73, 78, 255),
        )
        for checker_y in range(tile_size):
            for checker_x in range(tile_size):
                if (checker_x // 10 + checker_y // 10) % 2:
                    draw.point(
                        (x + checker_x, y + checker_y),
                        fill=(66, 67, 72, 255),
                    )

        tile = Image.open(io.BytesIO(base64.b64decode(texture["base64_png"]))).convert("RGBA")
        tile = tile.resize((tile_size, tile_size), Image.Resampling.NEAREST)
        sheet.alpha_composite(tile, (x, y))
        draw.text(
            (x, y + tile_size + 4),
            f"Variant {texture['variant_index']}",
            fill=(225, 225, 229, 255),
        )

    return sheet


def _generate_for_ui(
    prompt: str,
    variants: int,
    seed: float,
    steps: int,
    guidance_scale: float,
) -> tuple[Any, Any]:
    if not seed.is_integer():
        raise ValueError("Seed must be an integer.")
    result = generate_textures(prompt, variants, int(seed), steps, guidance_scale)
    seeds = ", ".join(str(texture["seed"]) for texture in result["textures"])
    summary = f"Generated **{result['count']}** textures · seeds: `{seeds}`"
    return (
        gr.update(value=_make_preview_sheet(result["textures"]), visible=True),
        gr.update(value=summary, visible=True),
    )


with gr.Blocks(title="Texture-Craft") as demo:
    gr.Markdown(
        "# Texture-Craft\n"
        "Generate crisp, transparent 16×16 item textures."
    )
    with gr.Row(elem_id="workspace-row"):
        with gr.Column(scale=1, min_width=0, elem_classes="input-panel"):
            prompt_input = gr.Textbox(
                label="Describe your texture",
                placeholder="An emerald potion bottle with a cork stopper",
                lines=3,
                max_length=MAX_PROMPT_LENGTH,
            )
            variants_input = gr.Slider(
                1, MAX_VARIANTS, value=1, step=1, label="Variants"
            )
            with gr.Row():
                steps_input = gr.Slider(10, MAX_STEPS, value=30, step=1, label="Steps")
                guidance_input = gr.Slider(
                    3, 15, value=9, step=0.5, label="Guidance"
                )
            seed_input = gr.Number(value=-1, precision=0, label="Seed (-1 for random)")
            generate_button = gr.Button("Generate textures", variant="primary")
        with gr.Column(scale=1, min_width=0, elem_classes="preview-panel"):
            gr.Markdown("### Preview")
            preview_output = gr.Image(
                label="Texture previews",
                width="100%",
                height=180,
                type="pil",
                image_mode="RGBA",
                interactive=False,
                show_label=False,
                container=False,
                buttons=[],
                elem_id="texture-preview",
            )
            summary_output = gr.Markdown("Choose a prompt and generate to see your textures.")
    generate_button.click(
        _generate_for_ui,
        inputs=[prompt_input, variants_input, seed_input, steps_input, guidance_input],
        outputs=[preview_output, summary_output],
        api_name=False,
    )
    gr.api(
        generate_textures,
        api_name="generate_textures",
        api_description="Generate 1–8 transparent 16×16 item PNGs. Returns base64 PNG and data URI content.",
        concurrency_limit=1,
    )

if IS_HF_SPACE:
    # ZeroGPU's startup CUDA emulation permits placing the pipeline before allocation.
    TextureEngine.get_instance(cache_downloaded_model=False, device="cuda")


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        mcp_server=True,
        css="""
        #workspace-row {
            display: grid !important;
            grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
            align-items: start;
            gap: 1rem;
        }
        #workspace-row > .column {
            min-width: 0 !important;
            width: auto !important;
        }
        @media (max-width: 640px) {
            #workspace-row {
                grid-template-columns: minmax(0, 1fr);
            }
        }
        .input-panel, .preview-panel {
            border: 1px solid var(--border-color-primary);
            border-radius: 12px;
            padding: 18px !important;
            background: var(--block-background-fill);
        }
        #texture-preview img {
            image-rendering: pixelated;
            object-fit: contain !important;
        }
        #texture-preview .image-container {
            background-color: #292a2e;
            background-image:
                linear-gradient(45deg, #393a3f 25%, transparent 25%),
                linear-gradient(-45deg, #393a3f 25%, transparent 25%),
                linear-gradient(45deg, transparent 75%, #393a3f 75%),
                linear-gradient(-45deg, transparent 75%, #393a3f 75%);
            background-size: 24px 24px;
            background-position: 0 0, 0 12px, 12px -12px, -12px 0;
        }
        #texture-preview .empty {
            height: 132px !important;
            border-radius: 8px;
            color: #d1d2d6;
            background-color: #292a2e;
            background-image:
                linear-gradient(45deg, #393a3f 25%, transparent 25%),
                linear-gradient(-45deg, #393a3f 25%, transparent 25%),
                linear-gradient(45deg, transparent 75%, #393a3f 75%),
                linear-gradient(-45deg, transparent 75%, #393a3f 75%);
            background-size: 24px 24px;
            background-position: 0 0, 0 12px, 12px -12px, -12px 0;
        }
        #texture-preview .empty .icon {
            display: none;
        }
        #texture-preview .empty::after {
            content: "16x16 texture previews";
            font-size: 0.9rem;
        }
        """,
    )
