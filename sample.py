"""ファインチューン済みUNetで16x16アイテムテクスチャを生成する。

256pxで生成し、16x16ブロック平均で無劣化に16x16へ畳み、
学習時に使ったマゼンタ背景をキーイングしてアルファへ戻す。
"""

import argparse
import pathlib
import sys

import numpy as np
import torch
from PIL import Image

from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DPMSolverMultistepScheduler

import mcpixel

BASE = "stable-diffusion-v1-5/stable-diffusion-v1-5"
BG = np.array([255, 0, 255], dtype=np.int16)
# 学習時のcaption dropoutは空文字列を無条件経路として学ばせたので、生成時も空が正しい。
# 非空のnegativeに含まれる語は学習語彙(11,613語)にほぼ無く、CFGの引き算にノイズを足す。
# 実測: negativeを空にすると usb drive / cat が成立し、非空だとどのguidanceでも成立しない。
NEGATIVE = ""


def fold_to_tile(image: Image.Image, size: int = 16) -> Image.Image:
    """256px を size x size へ畳む。1論理ピクセル = (256/size)^2 画素の平均。"""
    a = np.asarray(image.convert("RGB"), dtype=np.float32)
    c = a.shape[0] // size
    return Image.fromarray(a.reshape(size, c, size, c, 3).mean(axis=(1, 3))
                           .round().clip(0, 255).astype(np.uint8))


def key_out(tile: Image.Image, tol: int = 100) -> Image.Image:
    """マゼンタ背景を透過に戻す。"""
    rgb = np.asarray(tile.convert("RGB"), dtype=np.int16)
    bg = np.abs(rgb - BG).sum(axis=2) <= tol
    out = np.dstack([rgb.astype(np.uint8),
                     np.where(bg, 0, 255).astype(np.uint8)])
    return Image.fromarray(out, "RGBA")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unet", default="", help="学習済みUNetのディレクトリ。空なら素のSD1.5")
    ap.add_argument("--out", default="out_ft")
    ap.add_argument("--prompt", action="append", default=None)
    ap.add_argument("--n", type=int, default=1, help="プロンプトごとの生成枚数")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=9.0)  # 実測で7〜12が良く3〜5は眠い
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--tile", type=int, default=16)
    ap.add_argument("--cols", type=int, default=0, help="コンタクトシートの列数。0で自動")
    args = ap.parse_args()

    prompts = args.prompt or [
        "diamond sword", "copper ingot", "steel pickaxe", "gold helmet",
        "iron boots", "ruby gem", "emerald axe", "blue potion",
        "golden apple", "magic wand", "silver key", "obsidian shovel",
    ]

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float32 if device == "cpu" else torch.float16
    print(f"[info] device={device} unet={args.unet or 'base SD1.5'}", file=sys.stderr)

    pipe = StableDiffusionPipeline.from_pretrained(
        BASE, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False)
    if args.unet:
        pipe.unet = UNet2DConditionModel.from_pretrained(args.unet, torch_dtype=dtype)
    pipe = pipe.to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    outdir = pathlib.Path(args.out)
    for sub in ("raw", "tile"):
        (outdir / sub).mkdir(parents=True, exist_ok=True)

    tiles, labels = [], []
    for i, prompt in enumerate(prompts):
        for k in range(args.n):
            gen = torch.Generator(device="cpu").manual_seed(args.seed + i * 1000 + k)
            im = pipe(prompt, negative_prompt=NEGATIVE, num_inference_steps=args.steps,
                      guidance_scale=args.guidance, height=args.res, width=args.res,
                      generator=gen).images[0]
            name = prompt.replace(" ", "_") + (f"_{k}" if args.n > 1 else "")
            im.save(outdir / "raw" / f"{name}.png")
            tile = key_out(fold_to_tile(im, args.tile))
            tile.save(outdir / "tile" / f"{name}.png")
            tiles.append(tile); labels.append(name)
            print(f"[gen] {name}", file=sys.stderr)

    cols = args.cols or min(6, len(tiles))
    mcpixel.contact_sheet(tiles, cols=cols, cell=96).save(outdir / "contact_sheet.png")
    # シート上の並び順を残す（複数シードだと目視で追えなくなるため）
    outdir.joinpath("index.txt").write_text(
        "\n".join(f"{i//cols}行{i%cols+1}列  {n}" for i, n in enumerate(labels)), encoding="utf8")
    print(f"[done] {len(tiles)}枚 -> {outdir}", file=sys.stderr)


if __name__ == "__main__":
    main()
