"""16xModdedMinecraft からアイテムテクスチャだけを抜き、学習用データを作る。

キャプションはファイル名から作る。VLMは使わない:
  copper_sword.png -> "copper sword"
MOD固有の造語（低頻度語）はノイズなので落とす。

出力:
  images.npy   (N, 16, 16, 4) uint8
  captions.txt N行
"""

import argparse
import collections
import io
import json
import pathlib
import re
import sys

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

# キャプションから落とす語（画像の見た目を説明していない）
STOPWORDS = {"png", "item", "items", "texture", "textures", "0", "1", "2", "3", "4",
             "5", "6", "7", "8", "9", "00", "01", "02", "03", "small", "big"}
# ファイル名に含まれたら丸ごと捨てる（UI部品・アニメ定義など）
DROP_PATTERNS = re.compile(
    r"(gui|slot|widget|button|font|cursor|bar_|_bar|scroll|tooltip|background|"
    r"overlay|mask|blank|empty|placeholder|missing|debug|test_|_mcmeta)")


def tokenize(file_name: str) -> list[str]:
    stem = file_name.removesuffix(".png").lower()
    parts = re.split(r"[_\-.\s]+", stem)
    return [p for p in parts if p and p not in STOPWORDS and not p.isdigit()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", required=True)
    ap.add_argument("--out", default="data")
    ap.add_argument("--min-token-freq", type=int, default=20,
                    help="この回数未満の語はMOD固有の造語とみなし捨てる")
    ap.add_argument("--min-opaque", type=float, default=0.08,
                    help="不透明画素の割合の下限")
    ap.add_argument("--min-colors", type=int, default=3)
    args = ap.parse_args()

    files = sorted(pathlib.Path(args.parquet_dir).glob("*.parquet"))
    print(f"[info] parquet {len(files)}件", file=sys.stderr)

    # --- 1周目: アイテムだけ読み、画素で重複除去しつつ語彙を数える ---
    vocab = collections.Counter()
    seen: set[bytes] = set()
    images: list[np.ndarray] = []
    names: list[str] = []
    owners: list[tuple[str, str]] = []   # (project_id, mod_slug)
    stats = collections.Counter()

    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(
                batch_size=16384, columns=["file_name", "type", "image",
                                           "project_id", "mod_slug"]):
            d = batch.to_pydict()
            for fn, tp, rec, pid, slug in zip(d["file_name"], d["type"], d["image"],
                                              d["project_id"], d["mod_slug"]):
                stats["total"] += 1
                if tp != "item":
                    stats["block除外"] += 1
                    continue
                if DROP_PATTERNS.search(fn.lower()):
                    stats["UI等除外"] += 1
                    continue
                a = np.asarray(Image.open(io.BytesIO(rec["bytes"])).convert("RGBA"),
                               dtype=np.uint8)
                if a.shape != (16, 16, 4):
                    stats["サイズ違い除外"] += 1
                    continue
                key = a.tobytes()
                if key in seen:
                    stats["重複除外"] += 1
                    continue
                opaque = a[:, :, 3] > 16
                if opaque.mean() < args.min_opaque:
                    stats["ほぼ透明除外"] += 1
                    continue
                if len(np.unique(a[opaque][:, :3], axis=0)) < args.min_colors:
                    stats["単色除外"] += 1
                    continue
                toks = tokenize(fn)
                if not toks:
                    stats["名前なし除外"] += 1
                    continue
                seen.add(key)
                images.append(a)
                names.append(fn)
                owners.append((pid or "", slug or ""))
                vocab.update(toks)
            if stats["total"] % 200000 < 16384:
                print(f"  ... {stats['total']:,}件走査 / 採用 {len(images):,}", file=sys.stderr)

    # --- 2周目: MOD名由来の語を特定する ---
    # 高頻度なのに出現がひとつのMODに偏っている語は、素材名ではなくMODのブランド名。
    hot = {w for w, c in vocab.items() if c >= 50}
    by_token = collections.defaultdict(collections.Counter)
    for fn, (pid, _) in zip(names, owners):
        for t in tokenize(fn):
            if t in hot:
                by_token[t][pid] += 1
    branding = {t for t, c in by_token.items()
                if c.most_common(1)[0][1] / sum(c.values()) >= 0.9}
    print(f"[info] MOD固有語として除去: {len(branding)}語 "
          f"例={sorted(branding, key=lambda t: -vocab[t])[:10]}", file=sys.stderr)

    known = {w for w, c in vocab.items() if c >= args.min_token_freq} - branding
    captions = []
    keep = []
    for i, fn in enumerate(names):
        # その行のMOD名に含まれる語も落とす（slugが "tinkers-construct" なら tinkers/construct）
        slug_toks = set(re.split(r"[-_.\s]+", owners[i][1].lower()))
        toks = [t for t in tokenize(fn) if t in known and t not in slug_toks]
        # 同じ語の重複を潰す（copper_copper_ingot 対策）
        seen_t, uniq = set(), []
        for t in toks:
            if t not in seen_t:
                seen_t.add(t); uniq.append(t)
        if not uniq:
            stats["キャプション空除外"] += 1
            continue
        captions.append(" ".join(uniq))
        keep.append(i)

    arr = np.stack([images[i] for i in keep])
    outdir = pathlib.Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "images.npy", arr)
    (outdir / "captions.txt").write_text("\n".join(captions), encoding="utf8")
    (outdir / "stats.json").write_text(json.dumps({
        "kept": len(captions), "vocab_all": len(vocab), "vocab_kept": len(known),
        "filters": dict(stats)}, ensure_ascii=False, indent=2), encoding="utf8")

    print(f"\n[結果] 採用 {len(captions):,}枚  配列 {arr.nbytes/1e6:.0f}MB", file=sys.stderr)
    for k, v in stats.most_common():
        print(f"  {k}: {v:,}", file=sys.stderr)
    print(f"  語彙: 全{len(vocab):,}語 -> 採用{len(known):,}語", file=sys.stderr)
    cc = collections.Counter(captions)
    print(f"  ユニークなキャプション: {len(cc):,}", file=sys.stderr)
    print(f"  頻出キャプション: {cc.most_common(10)}", file=sys.stderr)

    # 目視確認用: ランダムに48枚をチェッカー背景で並べ、キャプションも書き出す
    rng = np.random.default_rng(0)
    idx = rng.choice(len(captions), size=min(48, len(captions)), replace=False)
    import mcpixel
    sheet = mcpixel.contact_sheet([Image.fromarray(arr[i]) for i in idx], cols=8, cell=64)
    sheet.save(outdir / "sample_sheet.png")
    (outdir / "sample_captions.txt").write_text(
        "\n".join(f"{n:2d}. {captions[i]}" for n, i in enumerate(idx)), encoding="utf8")


if __name__ == "__main__":
    main()
