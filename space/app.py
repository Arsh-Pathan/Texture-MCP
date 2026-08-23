"""minecraft-item-16px を誰でも試せるようにするデモ。

このモデルは256pxで生成し、16x16画素ずつ畳んで初めて本来の出力になる。
透過はマゼンタ(255,0,255)で表現されているのでキーイングして戻す。
その後処理をここで済ませ、利用者にはそのまま使える16x16 RGBA PNGを渡す。

テキストからの生成に加えて、下絵から仕上げるimg2imgのタブを持つ。
このモデルが苦手なのは質感ではなく「見慣れない物の形」なので、
形だけこちらで与えると当たらなかったプロンプトが通るようになる。
"""

import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import spaces
import torch
from PIL import Image
from diffusers import (StableDiffusionPipeline, StableDiffusionImg2ImgPipeline,
                       DPMSolverMultistepScheduler)

MODEL = "yuuki14202028/minecraft-item-16px"
BG = np.array([255, 0, 255], dtype=np.int16)   # 学習時に透過を表していた色
BG_RGBA = (255, 0, 255, 255)

# ZeroGPUでは起動時にcudaへ載せるのが作法（@spaces.GPU の外でも動く）
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")
pipe.set_progress_bar_config(disable=True)

# 同じ重みを使い回す。from_pretrainedし直すとVRAMに二重に載る。
i2i = StableDiffusionImg2ImgPipeline(**pipe.components)
i2i.set_progress_bar_config(disable=True)


def to_tile(image: Image.Image, size: int = 16, tol: int = 100) -> Image.Image:
    """256px を size x size へ畳み、マゼンタ背景を透過に戻す。"""
    a = np.asarray(image.convert("RGB"), dtype=np.float32)
    c = a.shape[0] // size
    rgb = a.reshape(size, c, size, c, 3).mean(axis=(1, 3)).round().clip(0, 255).astype(np.int16)
    alpha = np.where(np.abs(rgb - BG).sum(axis=2) <= tol, 0, 255).astype(np.uint8)
    return Image.fromarray(np.dstack([rgb.astype(np.uint8), alpha]))


def to_init(image: Image.Image, size: int = 16, up: int = 16) -> Image.Image:
    """下絵を学習時と同じ形式に揃える。16x16へ落とし、透過はマゼンタにして16倍に戻す。

    単純に縮小すると細い線が消える。256pxのキャンバスに引いた4pxの線は16x16では
    1画素の1/4しか覆わず、平均すると背景に負けてしまう。そこで「そのセルに不透明な
    画素が一つでもあれば残す」ことにし、色はセル内の不透明画素だけで平均する。
    """
    im = image.convert("RGBA")
    w, h = im.size
    if w % size or h % size:                      # size分割できる大きさに整える
        im = im.resize((max(w // size, 1) * size, max(h // size, 1) * size),
                       Image.Resampling.NEAREST)
    a = np.asarray(im, dtype=np.float32)
    bh, bw = a.shape[0] // size, a.shape[1] // size
    a = a.reshape(size, bh, size, bw, 4)
    solid = a[..., 3] >= 128                      # 筆先のアンチエイリアスは数えない
    cnt = solid.sum(axis=(1, 3))
    rgb = (a[..., :3] * solid[..., None]).sum(axis=(1, 3)) / np.maximum(cnt, 1)[..., None]
    tile = np.where((cnt > 0)[..., None], rgb, BG.astype(np.float32))
    tile = tile.round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(tile).resize((size * up, size * up), Image.Resampling.NEAREST)


def checkered(tile: Image.Image, cell: int = 16, out: int = 256) -> Image.Image:
    """透過が分かるようにチェッカー柄の上に置いた確認用画像を作る。"""
    big = tile.resize((out, out), Image.Resampling.NEAREST)
    bgim = Image.new("RGBA", (out, out), (238, 238, 242, 255))
    for y in range(0, out, cell):
        for x in range(0, out, cell):
            if (x // cell + y // cell) % 2:
                bgim.paste((206, 206, 214, 255), (x, y, x + cell, y + cell))
    bgim.alpha_composite(big)
    return bgim.convert("RGB")


def editor_image(value):
    """ImageEditorの戻り値から絵を取り出す。描画結果が無ければ読み込んだ背景を使う。"""
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get("composite") or value.get("background")
    return value


def pack(images, prompt):
    """生成結果を16x16へ畳んで保存し、表示用の拡大版も作る。"""
    stem = "".join(ch if ch.isalnum() else "_" for ch in prompt)[:40] or "texture"
    previews, files = [], []
    tmp = Path(tempfile.mkdtemp())
    for i, im in enumerate(images):
        tile = to_tile(im)
        path = tmp / f"{stem}_{i}.png"
        tile.save(path)                       # 本物の16x16 RGBA
        files.append(str(path))
        previews.append(checkered(tile))      # 画面表示用
    return previews, files


def duration(prompt, count, steps, guidance, seed):
    return int(6 + count * steps * 0.12)


@spaces.GPU(duration=duration)
def generate(prompt, count, steps, guidance, seed):
    if not prompt.strip():
        raise gr.Error("プロンプトを入れてください。例: copper sword")
    seed = int(seed)
    gen = None if seed < 0 else torch.Generator(device="cpu").manual_seed(seed)
    images = pipe([prompt] * int(count), negative_prompt=[""] * int(count),
                  num_inference_steps=int(steps), guidance_scale=float(guidance),
                  height=256, width=256, generator=gen).images
    return pack(images, prompt)


def sketch_duration(sketch, prompt, count, strength, steps, guidance, seed):
    # img2imgが実際に回すのは steps*strength 回。text2imgと同じ式では過大申告になる。
    return int(6 + count * steps * float(strength) * 0.12)


@spaces.GPU(duration=sketch_duration)
def generate_from_sketch(sketch, prompt, count, strength, steps, guidance, seed):
    src = editor_image(sketch)
    if src is None:
        raise gr.Error("下絵を描くか、画像を読み込んでください。")
    if not prompt.strip():
        raise gr.Error("プロンプトを入れてください。例: steel wrench")
    seed = int(seed)
    gen = None if seed < 0 else torch.Generator(device="cpu").manual_seed(seed)
    images = i2i(prompt=prompt, image=to_init(src), strength=float(strength),
                 negative_prompt="", num_inference_steps=int(steps),
                 guidance_scale=float(guidance), num_images_per_prompt=int(count),
                 generator=gen).images
    return pack(images, prompt)


def parse_list(text, limit=12):
    """カンマか改行で区切られた素材名を取り出す。ZeroGPUの持ち時間があるので数は絞る。"""
    items = [t.strip() for t in text.replace("\n", ",").split(",")]
    return [t for t in items if t][:limit]


def tiers_duration(sketch, materials, noun, strength, steps, guidance, seed):
    return int(8 + len(parse_list(materials)) * steps * float(strength) * 0.12)


@spaces.GPU(duration=tiers_duration)
def generate_tier_set(sketch, materials, noun, strength, steps, guidance, seed):
    """1枚の原案から素材違いの一式を作る。variants.py の階層セットに当たる。"""
    src = editor_image(sketch)
    if src is None:
        raise gr.Error("原案を描くか、画像を読み込んでください。")
    mats = parse_list(materials)
    if not mats:
        raise gr.Error("素材を1つ以上入れてください。例: iron, golden, diamond")
    prompts = [f"{m} {noun}".strip() for m in mats]
    seed = int(seed)
    # シルエットを揃えたいので、全プロンプトに同じ初期ノイズを与える
    gens = None if seed < 0 else [torch.Generator(device="cpu").manual_seed(seed)
                                  for _ in prompts]
    init = to_init(src)
    images = i2i(prompt=prompts, image=[init] * len(prompts), strength=float(strength),
                 negative_prompt=[""] * len(prompts), num_inference_steps=int(steps),
                 guidance_scale=float(guidance), generator=gens).images
    previews, files = [], []
    for prompt, im in zip(prompts, images):
        one_preview, one_file = pack([im], prompt)
        previews += one_preview
        files += one_file
    return previews, files


DESCRIPTION = """
# Minecraft Item Textures 16px

プロンプトから **本物の16×16 Minecraftアイテムテクスチャ** を作ります。
「ピクセルアート風のイラスト」ではなく、リソースパックにそのまま入れられるRGBA PNGです。

表示は見やすいように拡大していますが、ダウンロードされるファイルは **16×16、透過付き** です。

<details>
<summary>仕組み</summary>

Stable Diffusionは16×16を直接生成できません。VAEが8倍圧縮するので16px画像は潜在2×2にしかならず、
UNetの3段ダウンサンプルが破綻します。そこで学習画像を「16×16をニアレストで16倍した256px」に固定し、
モデルが常にその格子でしか描かないようにしました。出力を16×16画素ずつ平均すれば無損失で畳めます。
透過はマゼンタ(255,0,255)として学習させ、生成後にキーイングして戻しています。

**シードによる当たり外れが大きい**モデルです。1枚で判断せず、複数枚出して選んでください。
</details>
"""

SKETCH_HELP = """
**下絵で「形」を、モデルに「陰影と質感」を任せます。** このモデルが外すのは質感ではなく
見慣れない物の形なので、形さえ与えればテキストだけでは当たらないプロンプトも通ります。

触るべきつまみは **strength** です。低いほど下絵に忠実、高いほどモデルが描き直します。
0.45あたりだと下絵そのままの色合いが残り、0.9まで上げると下絵は構図の示唆くらいの扱いになります。
うまくいかないときは strength とシードを振ってください。
"""

TIERS_HELP = """
**1枚の原案から、素材違いの一式をまとめて作ります。** 素材名とアイテム名を組み合わせて
プロンプトを並べ、全部に同じ原案と同じシードを与えるので、**シルエットが揃った系列**になります。

strengthは0.5前後が目安です。上げすぎると素材ごとに形が散らばって「一式」に見えなくなります。

やっていることは「下絵から」と同じimg2imgで、違うのはプロンプトを複数まとめて回す点だけです。
"""

EXAMPLES = [
    ["copper sword", 4, 30, 9.0, 0],
    ["glass hammer", 4, 30, 9.0, 7],
    ["emerald key", 4, 30, 9.0, 3],
    ["coffee cup", 4, 30, 9.0, 11],
    ["glowing molten iron sword", 4, 30, 9.0, 5],
    ["tiny blue jellyfish", 4, 30, 9.0, 21],
]

with gr.Blocks(title="Minecraft Item Textures 16px") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Tabs():
        with gr.Tab("テキストから"):
            with gr.Row():
                with gr.Column(scale=2):
                    prompt = gr.Textbox(label="プロンプト", placeholder="copper sword",
                                        value="copper sword", lines=1)
                    run = gr.Button("生成", variant="primary")
                    with gr.Accordion("詳細設定", open=False):
                        count = gr.Slider(1, 8, value=4, step=1, label="枚数")
                        steps = gr.Slider(10, 50, value=30, step=1, label="ステップ数")
                        guidance = gr.Slider(1.0, 15.0, value=9.0, step=0.5,
                                             label="guidance（7〜12を推奨。3〜5は眠くなる）")
                        seed = gr.Number(value=0, precision=0, label="シード（-1でランダム）")
                with gr.Column(scale=3):
                    gallery = gr.Gallery(label="生成結果（表示は拡大）", columns=4,
                                         height=420, object_fit="contain")
                    files = gr.Files(label="16×16 PNG（透過付き）")

            run.click(generate, [prompt, count, steps, guidance, seed], [gallery, files])
            prompt.submit(generate, [prompt, count, steps, guidance, seed], [gallery, files])
            gr.Examples(EXAMPLES, [prompt, count, steps, guidance, seed])

        with gr.Tab("下絵から"):
            gr.Markdown(SKETCH_HELP)
            with gr.Row():
                with gr.Column(scale=2):
                    sketch = gr.ImageEditor(
                        label="下絵（描くか、画像を読み込む）",
                        type="pil", image_mode="RGBA",
                        canvas_size=(256, 256), fixed_canvas=True,
                        layers=False, transforms=[], sources=("upload", "clipboard"),
                        brush=gr.Brush(default_size=24), eraser=gr.Eraser(default_size=24),
                        height=320)
                    s_prompt = gr.Textbox(label="プロンプト", placeholder="steel wrench",
                                          value="steel wrench", lines=1)
                    s_strength = gr.Slider(0.3, 0.95, value=0.6, step=0.05,
                                           label="strength（低い＝下絵に忠実 / 高い＝描き直す）")
                    s_run = gr.Button("下絵から生成", variant="primary")
                    with gr.Accordion("詳細設定", open=False):
                        s_count = gr.Slider(1, 8, value=4, step=1, label="枚数")
                        s_steps = gr.Slider(10, 50, value=40, step=1, label="ステップ数")
                        s_guidance = gr.Slider(1.0, 15.0, value=9.0, step=0.5, label="guidance")
                        s_seed = gr.Number(value=0, precision=0, label="シード（-1でランダム）")
                with gr.Column(scale=3):
                    s_gallery = gr.Gallery(label="生成結果（表示は拡大）", columns=4,
                                           height=420, object_fit="contain")
                    s_files = gr.Files(label="16×16 PNG（透過付き）")

            s_inputs = [sketch, s_prompt, s_count, s_strength, s_steps, s_guidance, s_seed]
            s_run.click(generate_from_sketch, s_inputs, [s_gallery, s_files])
            s_prompt.submit(generate_from_sketch, s_inputs, [s_gallery, s_files])

        with gr.Tab("一式を作る"):
            gr.Markdown(TIERS_HELP)
            with gr.Row():
                with gr.Column(scale=2):
                    t_sketch = gr.ImageEditor(
                        label="原案（描くか、テクスチャを読み込む）",
                        type="pil", image_mode="RGBA",
                        canvas_size=(256, 256), fixed_canvas=True,
                        layers=False, transforms=[], sources=("upload", "clipboard"),
                        brush=gr.Brush(default_size=24), eraser=gr.Eraser(default_size=24),
                        height=320)
                    t_materials = gr.Textbox(
                        label="素材（カンマ区切り、最大12）",
                        value="wooden, stone, iron, golden, diamond, netherite, emerald, ruby",
                        lines=2)
                    t_noun = gr.Textbox(label="アイテム名", value="sword", lines=1)
                    t_strength = gr.Slider(0.3, 0.95, value=0.5, step=0.05,
                                           label="strength（0.5前後だとシルエットが揃う）")
                    t_run = gr.Button("一式を生成", variant="primary")
                    with gr.Accordion("詳細設定", open=False):
                        t_steps = gr.Slider(10, 50, value=40, step=1, label="ステップ数")
                        t_guidance = gr.Slider(1.0, 15.0, value=9.0, step=0.5, label="guidance")
                        t_seed = gr.Number(value=77, precision=0,
                                           label="シード（全素材で共有。-1でランダム）")
                with gr.Column(scale=3):
                    t_gallery = gr.Gallery(label="生成結果（素材の順）", columns=4,
                                           height=420, object_fit="contain")
                    t_files = gr.Files(label="16×16 PNG（透過付き）")

            t_run.click(generate_tier_set,
                        [t_sketch, t_materials, t_noun, t_strength, t_steps, t_guidance, t_seed],
                        [t_gallery, t_files])

    gr.Markdown(
        "モデル: [yuuki14202028/minecraft-item-16px]"
        "(https://huggingface.co/yuuki14202028/minecraft-item-16px) ・ "
        "コード: [github.com/yuuki14202028/pixelgen](https://github.com/yuuki14202028/pixelgen)\n\n"
        "学習データは235種のライセンスが混在するMOD由来のテクスチャです。"
        "生成物を再配布する場合は独自性をご自身で確認してください。"
    )

# SSRを有効にするとGalleryの更新がフロントに届かないことがあるので切る
demo.queue(max_size=20).launch(ssr_mode=False)
