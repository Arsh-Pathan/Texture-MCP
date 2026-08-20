# RunPodでの学習手順

SD1.5のUNetを、Minecraft MODアイテムテクスチャ369,342枚でフルファインチューンする。

## 1. Pod

- GPU: **RTX 4090 (Community Cloud, $0.34/h)**
- テンプレート: PyTorch 2.x / CUDA 12.x
- **ディスク 60GB以上**（SD1.5 5GB + 潜在キャッシュ3GB + データ0.4GB + チェックポイント約10GB）

## 2. セットアップ

```bash
pip install -U diffusers transformers accelerate safetensors bitsandbytes
```

## 3. データ転送（384MB）

ローカルから `data/images.npy` と `data/captions.txt` を送る。`latents.npy` は
Pod側で生成するので送らないこと。

```bash
# ローカル側
runpodctl send data/images.npy data/captions.txt
# Pod側で表示されたコードを使って
runpodctl receive <code>
```

`train.py` `sample.py` `mcpixel.py` も同様に転送する。

## 4. 学習

```bash
python train.py --data data --out runs/mc-item --bs 16 --lr 1e-5 --epochs 4
```

初回は潜在キャッシュ（VAEで369,342枚をエンコード、4090で15〜30分、3.0GB）が
先に走る。以降のエポックはこれを使い回すので再計算されない。

- 想定: **92,335ステップ / 40〜60 img/s / 7〜10時間 / $2.4〜3.4**
- ログは `runs/mc-item/log.jsonl`（step, loss, lr, img/s, 残り時間）
- チェックポイントは2,000ステップごと。直近2つだけ残す

Podが落ちたら再開する:

```bash
python train.py --data data --out runs/mc-item --bs 16 --epochs 4 \
  --resume runs/mc-item/step-0032000
```

## 5. 途中経過の確認

学習を止めずに、別シェルで保存済みチェックポイントから生成する。

```bash
python sample.py --unet runs/mc-item/step-0020000 --out preview
```

`preview/contact_sheet.png` を見て、16x16らしさとマゼンタ背景が
抜けているかを確認する。

## 6. 持ち帰り

UNetだけで3.4GB。オプティマイザ状態(`state.pt`)は不要なので送らない。

```bash
runpodctl send runs/mc-item/step-0092000/diffusion_pytorch_model.safetensors \
               runs/mc-item/step-0092000/config.json
```

ローカルでは:

```bash
.venv/bin/python sample.py --unet <落としたディレクトリ> --out out_ft
```

## 設計メモ

- **256pxで学習する**: 16x16をニアレストで x16 に拡大したものだけを見せるため、
  出力の格子が定義上16x16に揃う。16x16ブロック平均で無劣化に畳める。
  512pxでも同じことはできるが計算量が6.2倍になる（実測 0.68 img/s vs 4.2 img/s）。
- **背景はマゼンタ(255,0,255)**: 実データでこの色に近い不透明画素は0.057%、
  含む画像も0.55%しかない。黒(23%)や白(25%)は輪郭線と衝突するので使えない。
- **左右反転はしない**: Minecraftの道具は左下から右上へ向ける慣習があり、
  反転すると規約が壊れる。
- **キャプションを10%落とす**: classifier-free guidanceを効かせるため、
  無条件生成の経路を残しておく。
- **min-SNR重み付け (gamma=5)**: 収束を早める。
