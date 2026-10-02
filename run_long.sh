#!/usr/bin/env bash
# Next experiment #1 from CLAUDE.md: LONGER phase 2. Same dense parent (runs/dense/ckpt.pkl),
# same --data-seed 1234 batches for both arms, but 5000 steps each instead of 1500.
# Question: does the MoE-vs-dense gap keep growing or saturate?
# New run names (runs/ results are never overwritten). Idempotent + resumable like run_experiment.sh:
#     MAX_SECONDS=600 bash run_long.sh     # <=600 s of work per phase per call; rerun to continue
set -euo pipefail
cd "$(dirname "$0")"
[[ -x .venv/bin/python ]] && PY=.venv/bin/python || PY=python3
LONG_STEPS=${LONG_STEPS:-5000}
E=${E:-4}; K=${K:-2}
MAX_SECONDS=${MAX_SECONDS:-0}
export PYTHONUNBUFFERED=1
[[ -f runs/dense/DONE ]] || { echo "runs/dense not finished; run run_experiment.sh first"; exit 1; }

phase() {
  local name=$1; shift
  [[ -f runs/$name/DONE ]] && return 0
  echo "== phase: $name"
  $PY -m moe.train --run "runs/$name" --resume --max-seconds "$MAX_SECONDS" "$@"
  [[ -f runs/$name/DONE ]] || { echo "(paused: $name)"; exit 0; }
}

phase dense_cont_long --init runs/dense/ckpt.pkl --steps "$LONG_STEPS" --lr 3e-4 --lr-final 3e-5 --warmup 50
phase moe_long        --init runs/dense/ckpt.pkl --upcycle --n-experts "$E" --top-k "$K" \
                      --steps "$LONG_STEPS" --lr 3e-4 --lr-final 3e-5 --warmup 50
echo "LONG PHASE 2 DONE — compare runs/dense_cont_long/log.csv vs runs/moe_long/log.csv"
