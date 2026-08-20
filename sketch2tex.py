"""16x16のラフスケッチを下絵にして、モデルにMinecraft風テクスチャへ仕上げさせる。

img2img（SDEdit）を使う。学習も追加モデルも不要で、いまの重みのまま動く。
下絵で「形と配置」を決め、モデルに「陰影と質感」を任せる分担になる。
"""

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


SKETCHES = {
    "steel wrench": sketch_wrench,
    "wooden guitar": sketch_guitar,
    "golden key": sketch_key,
    "red umbrella": sketch_umbrella,
}
STRENGTHS = [0.45, 0.6, 0.75, 0.9]


def main() -> None:
    outdir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "out_sketch")
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
    for prompt, fn in SKETCHES.items():
        s16 = fn()
        s16.save(outdir / "sketch" / f"{prompt.replace(' ', '_')}.png")
        init = s16.resize((16 * UPSCALE, 16 * UPSCALE), Image.Resampling.NEAREST)
        # 1列目に下絵そのものを置いて、どこから始まったか分かるようにする
        rows.append(key_out(s16.convert("RGB"))); labels.append(f"{prompt} (下絵)")
        for st in STRENGTHS:
            gen = torch.Generator(device="cpu").manual_seed(77)
            im = pipe(prompt=prompt, image=init, strength=st, guidance_scale=9.0,
                      negative_prompt=NEGATIVE, num_inference_steps=40, generator=gen).images[0]
            tile = key_out(fold_to_tile(im, 16))
            tile.save(outdir / "tile" / f"{prompt.replace(' ', '_')}_s{st}.png")
            rows.append(tile); labels.append(f"{prompt} s={st}")
        print(f"[gen] {prompt}", file=sys.stderr, flush=True)

    cols = 1 + len(STRENGTHS)
    mcpixel.contact_sheet(rows, cols=cols, cell=96).save(outdir / "contact_sheet.png")
    outdir.joinpath("index.txt").write_text(
        f"列 = 下絵, strength {STRENGTHS}\n行 = {list(SKETCHES)}\n", encoding="utf8")
    print(f"[done] -> {outdir}", file=sys.stderr)


if __name__ == "__main__":
    main()
