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
        raise gr.Error("Enter a prompt. For example: copper sword")
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
        raise gr.Error("Draw something, or load an image.")
    if not prompt.strip():
        raise gr.Error("Enter a prompt. For example: steel wrench")
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
        raise gr.Error("Draw a design, or load an image.")
    mats = parse_list(materials)
    if not mats:
        raise gr.Error("Enter at least one material. For example: iron, golden, diamond")
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


# 画面の文言。ラベルは英語で固定し、長い説明だけブラウザの言語に合わせて出し分ける。
# gr.Markdown の value は I18n が確実に効くが、label 側は保証がないため使わない。
DESC_EN = """
# Minecraft Item Textures 16px

Makes **real 16×16 Minecraft item textures** from a prompt. Not "an illustration that looks like
pixel art" — an actual RGBA sprite you can drop straight into a resource pack.

The preview is enlarged so you can see it. The files you download are **16×16, with transparency**.

<details>
<summary>How it works</summary>

Stable Diffusion cannot generate a 16×16 image. Its VAE compresses by 8×, so 16px becomes a 2×2
latent and the UNet's three downsampling stages break. So every training image was a 16×16 texture
upscaled ×16 with nearest-neighbour, and the model only ever learned to draw on that grid.
Averaging each 16×16 block of the output recovers the sprite exactly. Transparency was trained as
magenta (255, 0, 255) and is keyed out afterwards.

**Seed variance is large.** Do not judge this model from one image — generate several and pick.
</details>
"""

DESC_JA = """
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

SKETCH_EN = """
**You give the shape, the model gives shading and material.** What this model gets wrong is the
*shape* of unfamiliar objects, not texture — so supplying the shape yourself makes prompts land
that text alone never does.

The dial that matters is **strength**. Low keeps your drawing, high lets the model redraw it.
Around 0.45 your own colours survive; by 0.9 the drawing is treated as a loose suggestion.
When a result misses, sweep strength and the seed before giving up.
"""

SKETCH_JA = """
**下絵で「形」を、モデルに「陰影と質感」を任せます。** このモデルが外すのは質感ではなく
見慣れない物の形なので、形さえ与えればテキストだけでは当たらないプロンプトも通ります。

触るべきつまみは **strength** です。低いほど下絵に忠実、高いほどモデルが描き直します。
0.45あたりだと下絵そのままの色合いが残り、0.9まで上げると下絵は構図の示唆くらいの扱いになります。
うまくいかないときは strength とシードを振ってください。
"""

TIERS_EN = """
**One design in, a whole set of material variants out.** Each material is combined with the item
name into a prompt, and every prompt gets the same design and the same seed — which is what keeps
the **silhouettes lined up** across the set.

Strength around 0.5 is the sweet spot. Push it higher and each material drifts into its own shape,
and it stops reading as a set.

Underneath this is the same img2img as the sketch tab. The only difference is running several
prompts at once.
"""

TIERS_JA = """
**1枚の原案から、素材違いの一式をまとめて作ります。** 素材名とアイテム名を組み合わせて
プロンプトを並べ、全部に同じ原案と同じシードを与えるので、**シルエットが揃った系列**になります。

strengthは0.5前後が目安です。上げすぎると素材ごとに形が散らばって「一式」に見えなくなります。

やっていることは「下絵から」と同じimg2imgで、違うのはプロンプトを複数まとめて回す点だけです。
"""

FOOTER_EN = """
Model: [yuuki14202028/minecraft-item-16px](https://huggingface.co/yuuki14202028/minecraft-item-16px)
 · Code: [github.com/yuuki14202028/pixelgen](https://github.com/yuuki14202028/pixelgen)

The training data is mod textures carrying 235 different licence values, including
all-rights-reserved. If you redistribute what you generate, verify its originality yourself.
"""

FOOTER_JA = """
モデル: [yuuki14202028/minecraft-item-16px](https://huggingface.co/yuuki14202028/minecraft-item-16px)
 ・ コード: [github.com/yuuki14202028/pixelgen](https://github.com/yuuki14202028/pixelgen)

学習データは235種のライセンスが混在するMOD由来のテクスチャです。
生成物を再配布する場合は独自性をご自身で確認してください。
"""

i18n = gr.I18n(
    en={"desc": DESC_EN, "sketch": SKETCH_EN, "tiers": TIERS_EN, "footer": FOOTER_EN},
    ja={"desc": DESC_JA, "sketch": SKETCH_JA, "tiers": TIERS_JA, "footer": FOOTER_JA},
)

EXAMPLES = [
    ["copper sword", 4, 30, 9.0, 0],
    ["glass hammer", 4, 30, 9.0, 7],
    ["emerald key", 4, 30, 9.0, 3],
    ["coffee cup", 4, 30, 9.0, 11],
    ["glowing molten iron sword", 4, 30, 9.0, 5],
    ["tiny blue jellyfish", 4, 30, 9.0, 21],
]

with gr.Blocks(title="Minecraft Item Textures 16px") as demo:
    gr.Markdown(i18n("desc"))

    with gr.Tabs():
        with gr.Tab("From text"):
            with gr.Row():
                with gr.Column(scale=2):
                    prompt = gr.Textbox(label="Prompt", placeholder="copper sword",
                                        value="copper sword", lines=1)
                    run = gr.Button("Generate", variant="primary")
                    with gr.Accordion("Advanced", open=False):
                        count = gr.Slider(1, 8, value=4, step=1, label="Images")
                        steps = gr.Slider(10, 50, value=30, step=1, label="Steps")
                        guidance = gr.Slider(1.0, 15.0, value=9.0, step=0.5,
                                             label="Guidance (7–12 works; 3–5 goes mushy)")
                        seed = gr.Number(value=0, precision=0, label="Seed (-1 for random)")
                with gr.Column(scale=3):
                    gallery = gr.Gallery(label="Results (shown enlarged)", columns=4,
                                         height=420, object_fit="contain")
                    files = gr.Files(label="16×16 PNG, with transparency")

            run.click(generate, [prompt, count, steps, guidance, seed], [gallery, files])
            prompt.submit(generate, [prompt, count, steps, guidance, seed], [gallery, files])
            gr.Examples(EXAMPLES, [prompt, count, steps, guidance, seed])

        with gr.Tab("From a drawing"):
            gr.Markdown(i18n("sketch"))
            with gr.Row():
                with gr.Column(scale=2):
                    sketch = gr.ImageEditor(
                        label="Drawing (draw here, or load an image)",
                        type="pil", image_mode="RGBA",
                        canvas_size=(256, 256), fixed_canvas=True,
                        layers=False, transforms=[], sources=("upload", "clipboard"),
                        brush=gr.Brush(default_size=24), eraser=gr.Eraser(default_size=24),
                        height=320)
                    s_prompt = gr.Textbox(label="Prompt", placeholder="steel wrench",
                                          value="steel wrench", lines=1)
                    s_strength = gr.Slider(0.3, 0.95, value=0.6, step=0.05,
                                           label="Strength (low = keep the drawing / high = redraw)")
                    s_run = gr.Button("Generate from drawing", variant="primary")
                    with gr.Accordion("Advanced", open=False):
                        s_count = gr.Slider(1, 8, value=4, step=1, label="Images")
                        s_steps = gr.Slider(10, 50, value=40, step=1, label="Steps")
                        s_guidance = gr.Slider(1.0, 15.0, value=9.0, step=0.5, label="Guidance")
                        s_seed = gr.Number(value=0, precision=0, label="Seed (-1 for random)")
                with gr.Column(scale=3):
                    s_gallery = gr.Gallery(label="Results (shown enlarged)", columns=4,
                                           height=420, object_fit="contain")
                    s_files = gr.Files(label="16×16 PNG, with transparency")

            s_inputs = [sketch, s_prompt, s_count, s_strength, s_steps, s_guidance, s_seed]
            s_run.click(generate_from_sketch, s_inputs, [s_gallery, s_files])
            s_prompt.submit(generate_from_sketch, s_inputs, [s_gallery, s_files])

        with gr.Tab("A whole set"):
            gr.Markdown(i18n("tiers"))
            with gr.Row():
                with gr.Column(scale=2):
                    t_sketch = gr.ImageEditor(
                        label="Design (draw here, or load a texture)",
                        type="pil", image_mode="RGBA",
                        canvas_size=(256, 256), fixed_canvas=True,
                        layers=False, transforms=[], sources=("upload", "clipboard"),
                        brush=gr.Brush(default_size=24), eraser=gr.Eraser(default_size=24),
                        height=320)
                    t_materials = gr.Textbox(
                        label="Materials (comma-separated, up to 12)",
                        value="wooden, stone, iron, golden, diamond, netherite, emerald, ruby",
                        lines=2)
                    t_noun = gr.Textbox(label="Item name", value="sword", lines=1)
                    t_strength = gr.Slider(0.3, 0.95, value=0.5, step=0.05,
                                           label="Strength (around 0.5 keeps silhouettes aligned)")
                    t_run = gr.Button("Generate the set", variant="primary")
                    with gr.Accordion("Advanced", open=False):
                        t_steps = gr.Slider(10, 50, value=40, step=1, label="Steps")
                        t_guidance = gr.Slider(1.0, 15.0, value=9.0, step=0.5, label="Guidance")
                        t_seed = gr.Number(value=77, precision=0,
                                           label="Seed (shared by every material; -1 for random)")
                with gr.Column(scale=3):
                    t_gallery = gr.Gallery(label="Results (in material order)", columns=4,
                                           height=420, object_fit="contain")
                    t_files = gr.Files(label="16×16 PNG, with transparency")

            t_run.click(generate_tier_set,
                        [t_sketch, t_materials, t_noun, t_strength, t_steps, t_guidance, t_seed],
                        [t_gallery, t_files])

    gr.Markdown(i18n("footer"))

# SSRを有効にするとGalleryの更新がフロントに届かないことがあるので切る
demo.queue(max_size=20).launch(ssr_mode=False, i18n=i18n)
