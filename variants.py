"""img2imgの使い所を4通り実演する。

A 階層セット   : 1つの原案から素材違いの系列を作る（シルエットが揃う）
B 既存資産の再解釈: 実データのテクスチャを種にして変種を作る
C ゆらぎ       : 同じプロンプトを低strengthで振り、微妙に違う個体を量産する
D 下絵から     : 16x16の落書きを仕上げさせる
"""

import io
import pathlib
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

from diffusers import (StableDiffusionPipeline, StableDiffusionImg2ImgPipeline,
                       UNet2DConditionModel, DPMSolverMultistepScheduler)
from sample import BASE, NEGATIVE, BG, fold_to_tile, key_out
import mcpixel

UP = 16
MAGENTA = tuple(int(v) for v in BG)


def to_init(tile16: Image.Image) -> Image.Image:
    """16x16 を学習時と同じ 256px へ。透過はマゼンタで潰す。"""
    rgba = tile16.convert("RGBA")
    bg = Image.new("RGB", rgba.size, MAGENTA)
    bg.paste(rgba, mask=rgba.split()[3])
    return bg.resize((16 * UP, 16 * UP), Image.Resampling.NEAREST)


def canvas() -> Image.Image:
    return Image.new("RGB", (16, 16), MAGENTA)


def sk_lantern():
    im = canvas(); d = ImageDraw.Draw(im)
    d.rectangle([5, 3, 10, 5], fill=(90, 90, 95))       # 蓋
    d.rectangle([5, 5, 10, 12], fill=(240, 210, 90))    # 光る本体
    d.rectangle([5, 12, 10, 13], fill=(90, 90, 95))     # 底
    d.line([(8, 1), (8, 3)], fill=(90, 90, 95))         # 吊り手
    return im


def sk_shield():
    im = canvas(); d = ImageDraw.Draw(im)
    d.polygon([(3, 2), (12, 2), (12, 9), (8, 14), (3, 9)], fill=(140, 60, 50))
    d.polygon([(5, 4), (10, 4), (10, 8), (8, 11), (5, 8)], fill=(200, 180, 90))
    return im


def sk_flask():
    im = canvas(); d = ImageDraw.Draw(im)
    d.rectangle([6, 2, 9, 6], fill=(200, 220, 230))     # 首
    d.polygon([(4, 7), (11, 7), (12, 14), (3, 14)], fill=(120, 200, 120))  # 胴
    d.rectangle([6, 1, 9, 2], fill=(140, 100, 60))      # コルク
    return im


def sk_anvil():
    im = canvas(); d = ImageDraw.Draw(im)
    d.rectangle([2, 3, 13, 6], fill=(70, 70, 78))       # 上面
    d.rectangle([5, 6, 10, 10], fill=(60, 60, 68))      # 胴
    d.rectangle([3, 10, 12, 13], fill=(70, 70, 78))     # 台
    return im


def sk_fish():
    im = canvas(); d = ImageDraw.Draw(im)
    d.ellipse([2, 5, 11, 11], fill=(80, 150, 200))
    d.polygon([(11, 8), (15, 4), (15, 12)], fill=(60, 120, 170))
    d.point((5, 7), fill=(20, 20, 20))
    return im


def sk_crown():
    im = canvas(); d = ImageDraw.Draw(im)
    d.polygon([(2, 12), (2, 5), (5, 8), (8, 4), (11, 8), (14, 5), (14, 12)],
              fill=(230, 190, 70))
    d.rectangle([2, 12, 14, 14], fill=(200, 160, 55))
    d.point((8, 10), fill=(200, 60, 70))
    return im


SKETCHES = {"iron lantern": sk_lantern, "wooden shield": sk_shield,
            "green potion flask": sk_flask, "steel anvil": sk_anvil,
            "blue fish": sk_fish, "golden crown": sk_crown}

TIERS = ["wooden sword", "stone sword", "iron sword", "golden sword",
         "diamond sword", "netherite sword", "emerald sword", "ruby sword"]


def main() -> None:
    out = pathlib.Path("out_variants")
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if device == "cpu" else torch.float16

    txt = StableDiffusionPipeline.from_pretrained(
        BASE, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False)
    txt.unet = UNet2DConditionModel.from_pretrained("models/mc-item-step92335", torch_dtype=dtype)
    txt = txt.to(device)
    txt.scheduler = DPMSolverMultistepScheduler.from_config(txt.scheduler.config)
    txt.set_progress_bar_config(disable=True)
    i2i = StableDiffusionImg2ImgPipeline(**txt.components)   # 重みを共有して二重確保を避ける
    i2i.set_progress_bar_config(disable=True)

    def gen_txt(prompt, seed):
        g = torch.Generator(device="cpu").manual_seed(seed)
        return txt(prompt, negative_prompt=NEGATIVE, num_inference_steps=30,
                   guidance_scale=9.0, height=256, width=256, generator=g).images[0]

    def gen_i2i(prompt, init, strength, seed, steps=40):
        g = torch.Generator(device="cpu").manual_seed(seed)
        return i2i(prompt=prompt, image=init, strength=strength, guidance_scale=9.0,
                   negative_prompt=NEGATIVE, num_inference_steps=steps, generator=g).images[0]

    def save(group, tiles, labels, cols):
        d = out / group
        (d / "tile").mkdir(parents=True, exist_ok=True)
        for t, n in zip(tiles, labels):
            t.save(d / "tile" / f"{n}.png")
        mcpixel.contact_sheet(tiles, cols=cols, cell=96).save(d / "contact_sheet.png")
        d.joinpath("index.txt").write_text("\n".join(
            f"{i//cols}行{i%cols+1}列  {n}" for i, n in enumerate(labels)), encoding="utf8")
        print(f"[save] {group}: {len(tiles)}枚", file=sys.stderr, flush=True)

    # --- A 階層セット -------------------------------------------------
    base = key_out(fold_to_tile(gen_txt("iron sword", 5), 16))
    init = to_init(base)
    tiles, labels = [base], ["00_原案(iron sword)"]
    for i, p in enumerate(TIERS):
        im = gen_i2i(p, init, 0.5, 300 + i)
        tiles.append(key_out(fold_to_tile(im, 16))); labels.append(f"{i+1:02d}_{p}")
    save("A_tier_set", tiles, labels, 3)

    # --- B 既存資産の再解釈 ---------------------------------------------
    arr = np.load("data/images.npy", mmap_mode="r")
    rng = np.random.default_rng(3)
    picks = [int(x) for x in rng.choice(len(arr), 4, replace=False)]
    tiles, labels = [], []
    for j, idx in enumerate(picks):
        src = Image.fromarray(np.array(arr[idx]))
        tiles.append(src); labels.append(f"{j}_元テクスチャ")
        init2 = to_init(src)
        for k, mod in enumerate(["mossy ancient", "golden royal", "cursed dark"]):
            im = gen_i2i(f"{mod} item", init2, 0.55, 400 + j * 10 + k)
            tiles.append(key_out(fold_to_tile(im, 16))); labels.append(f"{j}_{mod}")
    save("B_reinterpret", tiles, labels, 4)

    # --- C ゆらぎ ------------------------------------------------------
    base2 = key_out(fold_to_tile(gen_txt("magic staff", 21), 16))
    init3 = to_init(base2)
    tiles, labels = [base2], ["00_原案(magic staff)"]
    for i in range(8):
        im = gen_i2i("magic staff", init3, 0.32, 500 + i)
        tiles.append(key_out(fold_to_tile(im, 16))); labels.append(f"{i+1:02d}_ゆらぎ")
    save("C_jitter", tiles, labels, 3)

    # --- D 下絵から ----------------------------------------------------
    tiles, labels = [], []
    for prompt, fn in SKETCHES.items():
        s = fn()
        tiles.append(key_out(s)); labels.append(f"{prompt}_下絵")
        im = gen_i2i(prompt, s.resize((16 * UP, 16 * UP), Image.Resampling.NEAREST), 0.7, 600)
        tiles.append(key_out(fold_to_tile(im, 16))); labels.append(f"{prompt}_仕上げ")
    save("D_sketch", tiles, labels, 4)

    print("[done] -> out_variants", file=sys.stderr)


if __name__ == "__main__":
    main()
