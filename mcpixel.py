"""16x16 Minecraftテクスチャ用の共通後処理。

512pxで描かれたピクセルアートを、格子を検出して正しい解像度へ「戻す」ことと、
アイテムテクスチャの背景を透過にすることを担当する。
"""

from __future__ import annotations

import numpy as np
from PIL import Image

MC_SIZE = 16


def detect_grid(image: Image.Image, candidates=(8, 16, 24, 32, 48, 64)) -> int:
    """ピクセルアート画像のネイティブ格子サイズ（1辺のセル数）を推定する。

    格子が合っていればセル内の色分散がほぼ0になる性質を使う。
    """
    a = np.asarray(image.convert("RGB"), dtype=np.float32)
    h, w, _ = a.shape
    best, best_score = candidates[0], float("inf")
    for g in candidates:
        if h % g or w % g:
            continue
        b = a.reshape(g, h // g, g, w // g, 3)
        score = float(b.std(axis=(1, 3)).mean())
        # 十分小さくなった時点で採用（大きい格子ほどスコアは下がり続けるため）
        if score < best_score * 0.5:
            best, best_score = g, score
    return best


def to_tile(image: Image.Image, size: int = MC_SIZE) -> Image.Image:
    """ネイティブ格子で一度整えてから目標解像度へ落とす。

    512 -> 16 を一発でやるより、512 -> 32(ネイティブ) -> 16 の方が色が濁らない。
    """
    grid = detect_grid(image)
    im = image.convert("RGB").resize((grid, grid), Image.Resampling.BOX)
    if grid != size:
        im = im.resize((size, size), Image.Resampling.BOX)
    return im


def quantize(image: Image.Image, colors: int = 12) -> Image.Image:
    """本物のMCテクスチャに寄せるため色数を絞る（1テクスチャあたり10色前後）。"""
    return image.convert("RGB").quantize(colors=colors, method=Image.Quantize.MEDIANCUT,
                                         dither=Image.Dither.NONE).convert("RGB")


def cutout_background(image: Image.Image, tol: int = 40) -> Image.Image:
    """外周とつながった背景色（黒／白）を透過にする。アイテム用。

    透過付きで学習された素材は背景が黒く潰れて出てくるので、それを塩梅よく戻す。
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    h, w, _ = rgb.shape
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    bg = np.median(border, axis=0)
    if not (bg.max() < 60 or bg.min() > 200):  # 黒／白背景でなければ触らない
        return image.convert("RGBA")

    mask = (np.abs(rgb - bg).sum(axis=2) <= tol)
    # 外周から連結している領域だけを背景とみなす（内側の暗部を消さない）
    keep = np.zeros((h, w), bool)
    stack = [(y, x) for y in range(h) for x in (0, w - 1) if mask[y, x]]
    stack += [(y, x) for x in range(w) for y in (0, h - 1) if mask[y, x]]
    while stack:
        y, x = stack.pop()
        if keep[y, x] or not mask[y, x]:
            continue
        keep[y, x] = True
        stack += [(y + dy, x + dx) for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))
                  if 0 <= y + dy < h and 0 <= x + dx < w]

    out = image.convert("RGBA")
    a = np.asarray(out).copy()
    a[keep, 3] = 0
    return Image.fromarray(a)


def is_block(image: Image.Image) -> bool:
    """全面が埋まっていればブロック、外周に背景があればアイテムと判定する。"""
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    bg = np.median(border, axis=0)
    if not (bg.max() < 60 or bg.min() > 200):
        return True
    return float((np.abs(border - bg).sum(axis=1) <= 40).mean()) < 0.5


def contact_sheet(images: list[Image.Image], cols: int = 8, cell: int = 64,
                  checker: bool = True) -> Image.Image:
    """透過が分かるようにチェッカー背景の上に並べる。"""
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGBA", (cols * cell, rows * cell), (32, 32, 36, 255))
    if checker:
        c = cell // 8
        for y in range(0, sheet.height, c):
            for x in range(0, sheet.width, c):
                if (x // c + y // c) % 2:
                    sheet.paste((54, 54, 60, 255), (x, y, x + c, y + c))
    for i, im in enumerate(images):
        tile = im.convert("RGBA").resize((cell, cell), Image.Resampling.NEAREST)
        sheet.alpha_composite(tile, ((i % cols) * cell, (i // cols) * cell))
    return sheet
