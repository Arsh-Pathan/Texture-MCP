"""学習済みモデルの守備範囲を、テーマ別に広く探る。

50プロンプト x 2シード = 100枚。1シードでの当たり外れに騙されないよう、
すべて2シードずつ振る。テーマごとにコンタクトシートを分ける。
"""

import pathlib
import sys

import torch

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from sample import BASE, NEGATIVE, fold_to_tile, key_out
import mcpixel

GROUPS = {
    # 現代の道具。Minecraftにも学習データにもほぼ無いはず
    "01_modern": [
        "smartphone", "camera", "headphones", "laptop", "usb drive",
        "battery", "wristwatch", "eyeglasses", "compass", "paint brush",
    ],
    # 食べ物。データにある物と無い物が混在
    "02_food": [
        "ramen bowl", "pizza slice", "hamburger", "ice cream cone", "taco",
        "onigiri", "birthday cake", "popcorn", "pretzel", "boba tea",
    ],
    # 生き物。16x16で描くのが最も難しい部類
    "03_creature": [
        "cat", "frog", "octopus", "butterfly",
        "penguin", "snail", "dragon egg", "jellyfish",
    ],
    # 素材 x 形。学習データにまず存在しない掛け合わせ
    "04_material": [
        "diamond crown", "obsidian guitar", "emerald key", "gold skull",
        "amethyst clock", "copper telescope", "ruby heart", "glass sword",
    ],
    # 形容詞を重ねる。どこまで指示が通るか
    "05_complex": [
        "cracked ancient rune stone", "glowing molten iron sword",
        "frozen blue crystal axe", "tiny rusty golden gear",
        "burning red demon horn", "ancient mossy stone tablet",
    ],
    # 和物と抽象概念。形の無いものは描けるのか
    "06_abstract": [
        "katana", "shuriken", "torii gate", "sakura branch",
        "maneki neko", "lightning bolt", "black hole", "rainbow",
    ],
}

SEEDS = 2


def main() -> None:
    unet_dir = sys.argv[1] if len(sys.argv) > 1 else "models/mc-item-step92335"
    outroot = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "out_explore")

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float32 if device == "cpu" else torch.float16
    print(f"[info] device={device} unet={unet_dir}", file=sys.stderr)

    from diffusers import UNet2DConditionModel
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False)
    pipe.unet = UNet2DConditionModel.from_pretrained(unet_dir, torch_dtype=dtype)
    pipe = pipe.to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    total = 0
    for group, prompts in GROUPS.items():
        gdir = outroot / group
        (gdir / "tile").mkdir(parents=True, exist_ok=True)
        (gdir / "raw").mkdir(parents=True, exist_ok=True)
        tiles, labels = [], []
        for i, prompt in enumerate(prompts):
            for k in range(SEEDS):
                gen = torch.Generator(device="cpu").manual_seed(1000 + i * 17 + k)
                im = pipe(prompt, negative_prompt=NEGATIVE, num_inference_steps=30,
                          guidance_scale=9.0, height=256, width=256, generator=gen).images[0]
                name = f"{prompt.replace(' ', '_')}_{k}"
                im.save(gdir / "raw" / f"{name}.png")
                tile = key_out(fold_to_tile(im, 16))
                tile.save(gdir / "tile" / f"{name}.png")
                tiles.append(tile); labels.append(name)
                total += 1
            print(f"[gen] {group} {prompt}", file=sys.stderr, flush=True)
        cols = 6   # 2シードなので1行に3プロンプト
        mcpixel.contact_sheet(tiles, cols=cols, cell=96).save(gdir / "contact_sheet.png")
        gdir.joinpath("index.txt").write_text(
            "\n".join(f"{i//cols}行{i%cols+1}列  {n}" for i, n in enumerate(labels)),
            encoding="utf8")
    print(f"[done] {total}枚 -> {outroot}", file=sys.stderr)


if __name__ == "__main__":
    main()
