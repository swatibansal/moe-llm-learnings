#!/usr/bin/env bash
# Full experiment: dense -> (a) keep dense  vs  (b) upcycle to MoE, same data order.
#
# Idempotent + resumable: every phase writes runs/<name>/ckpt.pkl (full state) and
# runs/<name>/DONE when finished. Re-running this script picks up where it left off,
# so you can Ctrl-C at any time, or run it in time-boxed chunks:
#     MAX_SECONDS=150 bash run_experiment.sh      # do <=150 s of work per phase, then exit
# Phase 1 ~0.25 s/step on 4 CPU cores; MoE phase ~1 s/step (E=4 experts evaluated densely).
set -euo pipefail
cd "$(dirname "$0")"
DENSE_STEPS=${DENSE_STEPS:-2000}
CONT_STEPS=${CONT_STEPS:-1500}
E=${E:-4}; K=${K:-2}
MAX_SECONDS=${MAX_SECONDS:-0}
export PYTHONUNBUFFERED=1

phase() {  # name, then train args
  local name=$1; shift
  [[ -f runs/$name/DONE ]] && return 0
  echo "== phase: $name"
  python3 -m moe.train --run "runs/$name" --resume --max-seconds "$MAX_SECONDS" "$@"
  [[ -f runs/$name/DONE ]] || { echo "(paused: $name)"; exit 0; }
}

phase dense      --steps "$DENSE_STEPS" --lr 1e-3 --lr-final 3e-4 --warmup 100
phase dense_cont --init runs/dense/ckpt.pkl --steps "$CONT_STEPS" --lr 3e-4 --lr-final 3e-5 --warmup 50
phase moe        --init runs/dense/ckpt.pkl --upcycle --n-experts "$E" --top-k "$K" \
                 --steps "$CONT_STEPS" --lr 3e-4 --lr-final 3e-5 --warmup 50

python3 plot.py
echo "ALL DONE — see runs/loss_curves.png and runs/*/log.csv"
