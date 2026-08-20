"""SD1.5 の UNet を Minecraft MODアイテムテクスチャでフルファインチューンする。

前提:
  - 16x16 RGBA を ニアレスト x16 で 256px に拡大したものだけを学習する。
    こうすると出力の格子が定義上ちょうど 16x16 に揃い、32x32ブロック平均で
    無劣化に 16x16 へ畳める。
  - 透過はマゼンタ(255,0,255)で合成する。実データでこの色を含む画像は0.55%しか
    ないので、生成後にキーイングしてアルファへ戻せる。

VAEの潜在は事前計算してキャッシュする（4エポック回すなら4回エンコードするのは無駄）。
テキストは毎ステップCLIPに通す（UNetに比べれば誤差の範囲）。
"""

import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

BASE = "stable-diffusion-v1-5/stable-diffusion-v1-5"
BG = np.array([255, 0, 255], dtype=np.float32)   # キーイング用の背景色
UPSCALE = 16                                      # 16x16 -> 256px


# --------------------------------------------------------------------------- データ

def composite_and_upscale(rgba: np.ndarray) -> np.ndarray:
    """(B,16,16,4) uint8 -> (B,3,256,256) float32 [-1,1]"""
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    rgb = rgba[..., :3].astype(np.float32) * a + BG * (1.0 - a)
    big = rgb.repeat(UPSCALE, axis=1).repeat(UPSCALE, axis=2)
    return (big / 127.5 - 1.0).transpose(0, 3, 1, 2)


@torch.no_grad()
def cache_latents(images: np.ndarray, path: pathlib.Path, device: str, batch: int = 64):
    """全画像をVAEで潜在に変換して memmap に落とす。"""
    # bf16を使う。fp16と違い指数部がfp32と同じなのでSD1.5のVAEでも溢れない。
    vdtype = torch.bfloat16 if device == "cuda" else torch.float32
    vae = AutoencoderKL.from_pretrained(BASE, subfolder="vae", torch_dtype=vdtype).to(device).eval()
    n = len(images)
    side = 256 // 8
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(n, 4, side, side))
    t0 = time.time()
    for i in range(0, n, batch):
        x = torch.from_numpy(composite_and_upscale(images[i:i + batch])).to(device, vdtype)
        lat = vae.encode(x).latent_dist.mean * vae.config.scaling_factor
        if not torch.isfinite(lat).all():
            raise RuntimeError(f"潜在にNaN/Infが出ました (i={i})")
        out[i:i + batch] = lat.float().cpu().numpy().astype(np.float16)
        if i % (batch * 100) == 0:
            done = i + batch
            print(f"  latent {done:,}/{n:,}  {done/(time.time()-t0):.0f} img/s", flush=True)
    out.flush()
    del vae
    if device == "cuda":
        torch.cuda.empty_cache()
    print(f"[info] 潜在キャッシュ完了 {path} ({out.nbytes/1e9:.2f}GB)", flush=True)


class ItemDataset(Dataset):
    def __init__(self, latents_path: pathlib.Path, captions: list[str], dropout: float):
        self.path = latents_path
        self.latents = None          # ワーカープロセスごとに開く
        self.captions = captions
        self.dropout = dropout
        n = len(np.load(latents_path, mmap_mode="r"))
        assert n == len(captions), f"潜在{n}件とキャプション{len(captions)}件が不一致"

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, i):
        if self.latents is None:
            self.latents = np.load(self.path, mmap_mode="r")
        cap = self.captions[i]
        # classifier-free guidance を効かせるため一定割合を無条件にする
        if self.dropout and np.random.rand() < self.dropout:
            cap = ""
        return torch.from_numpy(np.array(self.latents[i])), cap


# --------------------------------------------------------------------------- 学習

def compute_snr(scheduler, timesteps):
    ac = scheduler.alphas_cumprod.to(timesteps.device)[timesteps].float()
    return ac / (1.0 - ac)


def lr_at(step: int, total: int, warmup: int, peak: float) -> float:
    if step < warmup:
        return peak * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs/mc-item")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=4.0)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--snr-gamma", type=float, default=5.0, help="min-SNR重み付け。0で無効")
    ap.add_argument("--caption-dropout", type=float, default=0.1)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--max-steps", type=int, default=0, help="動作確認用の打ち切り")
    ap.add_argument("--resume", default="", help="再開するチェックポイントのディレクトリ")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="先頭N枚だけ使う（動作確認用）")
    ap.add_argument("--vae-batch", type=int, default=64,
                    help="潜在キャッシュ時のバッチ。256pxのVAEは活性化が大きいので小さめに")
    ap.add_argument("--no-8bit", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[info] device={device} autocast={dtype}", flush=True)

    data = pathlib.Path(args.data)
    outdir = pathlib.Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    captions = [c for c in data.joinpath("captions.txt")
                .read_text(encoding="utf8").split("\n") if c.strip()]

    if args.limit:
        captions = captions[:args.limit]
    latents_path = data / (f"latents_{args.limit}.npy" if args.limit else "latents.npy")
    if not latents_path.exists():
        print("[info] 潜在キャッシュを作成します", flush=True)
        imgs = np.load(data / "images.npy", mmap_mode="r")
        cache_latents(imgs[:args.limit] if args.limit else imgs, latents_path, device, args.vae_batch)

    ds = ItemDataset(latents_path, captions, args.caption_dropout)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                    pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    total_steps = args.max_steps or int(len(ds) * args.epochs / args.bs)
    print(f"[info] {len(ds):,}枚 / {total_steps:,}ステップ (bs={args.bs})", flush=True)

    tokenizer = CLIPTokenizer.from_pretrained(BASE, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(BASE, subfolder="text_encoder",
                                                 torch_dtype=dtype).to(device).eval()
    text_encoder.requires_grad_(False)

    if args.resume:
        unet = UNet2DConditionModel.from_pretrained(args.resume, torch_dtype=torch.float32)
    else:
        unet = UNet2DConditionModel.from_pretrained(BASE, subfolder="unet",
                                                    torch_dtype=torch.float32)
    unet = unet.to(device)
    unet.enable_gradient_checkpointing()
    unet.train()

    scheduler = DDPMScheduler.from_pretrained(BASE, subfolder="scheduler")

    if args.no_8bit:
        opt = torch.optim.AdamW(unet.parameters(), lr=args.lr, weight_decay=1e-2)
    else:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(unet.parameters(), lr=args.lr, weight_decay=1e-2)

    start_step = 0
    if args.resume:
        st = pathlib.Path(args.resume) / "state.pt"
        if st.exists():
            blob = torch.load(st, map_location="cpu")
            opt.load_state_dict(blob["opt"]); start_step = blob["step"]
            print(f"[info] step {start_step} から再開", flush=True)

    log = outdir.joinpath("log.jsonl").open("a")
    step = start_step
    t0 = time.time(); seen = 0

    while step < total_steps:
        for latents, caps in dl:
            if step >= total_steps:
                break
            latents = latents.to(device, dtype)
            tok = tokenizer(list(caps), padding="max_length", max_length=77,
                            truncation=True, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                emb = text_encoder(tok)[0].to(dtype)

            noise = torch.randn_like(latents)
            t = torch.randint(0, scheduler.config.num_train_timesteps, (latents.shape[0],),
                              device=device, dtype=torch.long)
            noisy = scheduler.add_noise(latents.float(), noise.float(), t).to(dtype)

            for g in opt.param_groups:
                g["lr"] = lr_at(step, total_steps, args.warmup, args.lr)

            if device == "cuda":
                with torch.autocast(device_type="cuda", dtype=dtype):
                    pred = unet(noisy, t, encoder_hidden_states=emb).sample
            else:
                pred = unet(noisy, t, encoder_hidden_states=emb).sample
            loss = F.mse_loss(pred.float(), noise.float(), reduction="none").mean(dim=(1, 2, 3))
            if args.snr_gamma:
                snr = compute_snr(scheduler, t)
                loss = loss * (torch.clamp(snr, max=args.snr_gamma) / snr)
            loss = loss.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)

            step += 1; seen += latents.shape[0]
            if step % 50 == 0:
                rate = seen / (time.time() - t0)
                eta = (total_steps - step) * args.bs / max(rate, 1e-6) / 3600
                rec = {"step": step, "loss": round(loss.item(), 5),
                       "lr": round(opt.param_groups[0]["lr"], 9),
                       "img_s": round(rate, 1), "eta_h": round(eta, 2)}
                print(f"step {step:>7}/{total_steps}  loss {rec['loss']:.4f}  "
                      f"{rec['img_s']:.1f} img/s  残り {rec['eta_h']:.1f}h", flush=True)
                log.write(json.dumps(rec) + "\n"); log.flush()

            if step % args.save_every == 0 or step == total_steps:
                ck = outdir / f"step-{step:07d}"
                unet.save_pretrained(ck)
                torch.save({"opt": opt.state_dict(), "step": step}, ck / "state.pt")
                print(f"[save] {ck}", flush=True)
                for old in sorted(outdir.glob("step-*"))[:-2]:   # 直近2つだけ残す
                    for f in old.iterdir():
                        f.unlink()
                    old.rmdir()

    print(f"[done] {step}ステップ / {(time.time()-t0)/3600:.2f}時間", flush=True)


if __name__ == "__main__":
    main()
