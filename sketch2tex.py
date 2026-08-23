"""16x16のラフスケッチを下絵にして、モデルにMinecraft風テクスチャへ仕上げさせる。

img2img（SDEdit）を使う。学習も追加モデルも不要で、いまの重みのまま動く。
下絵で「形と配置」を決め、モデルに「陰影と質感」を任せる分担になる。
"""

import argparse
import pathlib
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

from diffusers import StableDiffusionImg2ImgPipeline, UNet2DConditionModel, DPMSolverMultistepScheduler
from sample import BASE, NEGATIVE, BG, fold_to_tile, key_out
import mcpixel

UPSCALE = 16   # 16x16 -> 256px


def canvas() -> Image.Image:
    """学習時と同じマゼンタ背景の16x16キャンバス。"""
    return Image.new("RGB", (16, 16), tuple(int(v) for v in BG))


def sketch_wrench() -> Image.Image:
    im = canvas(); d = ImageDraw.Draw(im)
    d.line([(4, 12), (11, 4)], fill=(150, 150, 158), width=2)   # 柄
    d.rectangle([2, 11, 6, 15], fill=(150, 150, 158))            # 口
    d.rectangle([3, 12, 5, 14], fill=tuple(int(v) for v in BG))  # 口の穴
    d.rectangle([10, 2, 14, 6], fill=(150, 150, 158))
    d.rectangle([11, 3, 13, 5], fill=tuple(int(v) for v in BG))
    return im


def sketch_guitar() -> Image.Image:
    im = canvas(); d = ImageDraw.Draw(im)
    d.ellipse([2, 8, 10, 15], fill=(150, 95, 45))                # 胴
    d.rectangle([9, 3, 11, 10], fill=(110, 70, 35))              # ネック
    d.rectangle([9, 1, 12, 3], fill=(90, 55, 25))                # ヘッド
    return im


def sketch_key() -> Image.Image:
    im = canvas(); d = ImageDraw.Draw(im)
    d.ellipse([2, 2, 8, 8], fill=(220, 180, 60))                 # 頭
    d.ellipse([4, 4, 6, 6], fill=tuple(int(v) for v in BG))
    d.line([(7, 7), (13, 13)], fill=(220, 180, 60), width=2)     # 軸
    d.rectangle([10, 13, 12, 15], fill=(220, 180, 60))           # 歯
    return im


def sketch_umbrella() -> Image.Image:
    im = canvas(); d = ImageDraw.Draw(im)
    d.pieslice([1, 2, 15, 14], 180, 360, fill=(190, 50, 50))     # 傘
    d.line([(8, 8), (8, 14)], fill=(120, 80, 40), width=1)       # 柄
    d.line([(8, 14), (11, 14)], fill=(120, 80, 40), width=1)
    return im


def load_sketch(path: pathlib.Path) -> Image.Image:
    """外部PNGを下絵として読む。透過部は学習時と同じマゼンタで埋める。"""
    im = Image.open(path).convert("RGBA")
    if im.size != (16, 16):
        im = im.resize((16, 16), Image.Resampling.NEAREST)
    return Image.alpha_composite(canvas().convert("RGBA"), im).convert("RGB")


SKETCHES = {
    "steel wrench": sketch_wrench,
    "wooden guitar": sketch_guitar,
    "golden key": sketch_key,
    "red umbrella": sketch_umbrella,
}
STRENGTHS = [0.45, 0.6, 0.75, 0.9]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="out_sketch")
    ap.add_argument("--sketch", action="append", default=None, metavar="PNG",
                    help="下絵にする16x16 PNG。省略時は組み込みの手描き4種")
    ap.add_argument("--prompt", action="append", default=None,
                    help="--sketchと同じ順で対応。省略分はファイル名から作る")
    ap.add_argument("--seed", action="append", type=int, default=None,
                    help="複数指定可。1シードでの判断は当てにならないので既定でも複数振る")
    ap.add_argument("--strength", action="append", type=float, default=None)
    args = ap.parse_args()

    strengths = args.strength or STRENGTHS
    seeds = args.seed or [77]
    if args.sketch:
        paths = [pathlib.Path(p) for p in args.sketch]
        given = args.prompt or []
        # プロンプト未指定分はファイル名から: skull_fragment.png -> "skull fragment"
        prompts = [given[i] if i < len(given) else p.stem.replace("_", " ")
                   for i, p in enumerate(paths)]
        sources = [(pr, (lambda q=p: load_sketch(q))) for pr, p in zip(prompts, paths)]
    else:
        sources = list(SKETCHES.items())

    outdir = pathlib.Path(args.out)
    (outdir / "sketch").mkdir(parents=True, exist_ok=True)
    (outdir / "tile").mkdir(parents=True, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if device == "cpu" else torch.float16

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        BASE, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False)
    pipe.unet = UNet2DConditionModel.from_pretrained("models/mc-item-step92335", torch_dtype=dtype)
    pipe = pipe.to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    rows, labels = [], []
    for prompt, fn in sources:
        stem = prompt.replace(" ", "_")
        s16 = fn()
        s16.save(outdir / "sketch" / f"{stem}.png")
        init = s16.resize((16 * UPSCALE, 16 * UPSCALE), Image.Resampling.NEAREST)
        for seed in seeds:
            # 各行の1列目に下絵そのものを置いて、どこから始まったか分かるようにする
            rows.append(key_out(s16.convert("RGB"))); labels.append(f"{prompt} (下絵)")
            for st in strengths:
                gen = torch.Generator(device="cpu").manual_seed(seed)
                im = pipe(prompt=prompt, image=init, strength=st, guidance_scale=9.0,
                          negative_prompt=NEGATIVE, num_inference_steps=40, generator=gen).images[0]
                tile = key_out(fold_to_tile(im, 16))
                tile.save(outdir / "tile" / f"{stem}_seed{seed}_s{st}.png")
                rows.append(tile); labels.append(f"{prompt} seed={seed} s={st}")
            print(f"[gen] {prompt} seed={seed}", file=sys.stderr, flush=True)

    cols = 1 + len(strengths)
    mcpixel.contact_sheet(rows, cols=cols, cell=96).save(outdir / "contact_sheet.png")
    outdir.joinpath("index.txt").write_text(
        "\n".join(f"{i//cols}行{i%cols+1}列  {n}" for i, n in enumerate(labels)) + "\n",
        encoding="utf8")
    print(f"[done] -> {outdir}", file=sys.stderr)


if __name__ == "__main__":
    main()
