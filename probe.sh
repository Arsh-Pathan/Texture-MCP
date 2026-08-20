#!/usr/bin/env bash
# 学習中のチェックポイントを同じ条件で検証し、out_<step>/ に取り込む。
#
#   ./probe.sh            # 最新チェックポイント
#   ./probe.sh 20000      # step-0020000 を指定
#
# シードとプロンプトは固定してあるので、別のステップ同士を直接比較できる。
# 注意: 2026-08-20にnegativeを空・guidanceを9へ変更した。それ以前に取った
# out_4000 / out_16000 / out_58000 / out_66000 / out_92335 とは生成条件が違う。
set -euo pipefail

POD_HOST="${POD_HOST:-64.119.209.250}"
POD_PORT="${POD_PORT:-19420}"
KEY="${KEY:-$HOME/.ssh/id_ed25519}"
SSH_OPTS=(-o StrictHostKeyChecking=no -i "$KEY")

remote() { ssh "${SSH_OPTS[@]}" -p "$POD_PORT" "root@$POD_HOST" "$@"; }

if [ $# -ge 1 ]; then
  CK="/workspace/runs/mc-item/step-$(printf '%07d' "$1")"
else
  CK=$(remote "ls -d /workspace/runs/mc-item/step-* | tail -1")
fi
STEP=$(basename "$CK" | sed 's/step-0*//')
echo "検証対象: $CK (step $STEP)"

# --- 1) 基本セット: バニラMinecraftにある素材と形 ---
remote "cd /workspace/mcgen && HF_HOME=/workspace/hf python sample.py --unet '$CK' \
  --out /workspace/probe_out/preview --steps 30 --guidance 9 --seed 0 --n 2 --cols 6 \
  --prompt 'copper sword' --prompt 'iron pickaxe' --prompt 'ruby gem' \
  --prompt 'golden helmet' --prompt 'diamond boots' --prompt 'steel ingot' \
  2>&1 | grep -acE '^\[gen\]'"

# --- 2) 診断セット: 未知の組み合わせ / Minecraftに無い概念 / 複雑な修飾 ---
#     1シードでは当たり外れを取り違えるので、各プロンプト3シード振る
remote "cd /workspace/mcgen && HF_HOME=/workspace/hf python sample.py --unet '$CK' \
  --out /workspace/probe_out/probe --steps 30 --guidance 9 --seed 4242 --n 3 --cols 6 \
  --prompt 'glass hammer' --prompt 'copper crown' \
  --prompt 'obsidian sword' --prompt 'emerald pickaxe' --prompt 'amethyst boots' \
  --prompt 'banana' --prompt 'sushi' --prompt 'coffee cup' --prompt 'umbrella' \
  --prompt 'syringe' --prompt 'donut' --prompt 'cheese' \
  --prompt 'microscope' --prompt 'guitar' --prompt 'telescope' --prompt 'violin' \
  --prompt 'cracked purple crystal sword' --prompt 'rusty steel gear' \
  --prompt 'tiny glowing green potion bottle' --prompt 'burning blue lantern' \
  2>&1 | grep -acE '^\[gen\]'"

# --- 3) シード依存を見るセット: 同じ語を4シードで振る ---
remote "cd /workspace/mcgen && HF_HOME=/workspace/hf python sample.py --unet '$CK' \
  --out /workspace/probe_out/steel --steps 30 --guidance 9 --seed 99 --n 4 --cols 4 \
  --prompt 'steel ingot' --prompt 'iron ingot' \
  2>&1 | grep -acE '^\[gen\]'"

OUT="out_$STEP"
mkdir -p "$OUT"
scp "${SSH_OPTS[@]}" -P "$POD_PORT" -q -r \
  "root@$POD_HOST:/workspace/probe_out/*" "$OUT/"
remote "rm -rf /workspace/probe_out"

"${PY:-.venv/bin/python}" - "$OUT" <<'PY'
import sys, glob, pathlib
from PIL import Image
out = sys.argv[1]
for group in ("preview", "probe", "steel"):
    files = sorted(glob.glob(f"{out}/{group}/tile/*.png"))
    if not files:
        continue
    import re, collections
    per = collections.defaultdict(lambda: [0, 0])
    for f in files:
        name = pathlib.Path(f).stem
        prompt = re.sub(r"_\d+$", "", name)
        a = Image.open(f).split()[3]
        per[prompt][1] += 1
        if sum(1 for p in a.get_flattened_data() if p == 0) >= 20:
            per[prompt][0] += 1
    keyed = sum(v[0] for v in per.values())
    print(f"  {group:8s} {len(files):3d}枚  背景が抜けた {keyed}/{len(files)}")
    dead = [k for k, v in per.items() if v[0] == 0]
    if dead:
        print(f"           全シードで抜けず: {', '.join(sorted(dead))}")
PY
echo "-> $OUT/ に取り込みました"
