#!/usr/bin/env bash
# Next experiment #2 from CLAUDE.md: sweep n_experts E in {2,4,8} x top_k in {1,2}.
# Every arm upcycles the SAME dense parent (runs/dense/ckpt.pkl) and trains 1500 steps with the
# phase-2 schedule and --data-seed 1234, so all are directly comparable to runs/dense_cont (control)
# and runs/moe (E=4, k=2 — already done, skipped here). Runs go to runs/moe_e{E}_k{K}.
# Key question: E=4, top-1 has the SAME active params as dense — is it still better than dense_cont?
# Idempotent + resumable like run_experiment.sh:  MAX_SECONDS=600 bash run_sweep.sh ; rerun to continue.
set -euo pipefail
cd "$(dirname "$0")"
[[ -x .venv/bin/python ]] && PY=.venv/bin/python || PY=python3
STEPS=${STEPS:-1500}
MAX_SECONDS=${MAX_SECONDS:-0}
GRID=${GRID:-"2:1 2:2 4:1 8:1 8:2"}   # E:K pairs; 4:2 is runs/moe
GATE_GRAD=${GATE_GRAD:-renorm}         # renorm (original) | stopgrad (router learns with top-1; see model.moe_ffn)
SUFFIX=${SUFFIX:-}                     # appended to run names, e.g. SUFFIX=_sg for the stopgrad rerun
export PYTHONUNBUFFERED=1
[[ -f runs/dense/DONE ]] || { echo "runs/dense not finished; run run_experiment.sh first"; exit 1; }

for ek in $GRID; do
  E=${ek%%:*}; K=${ek##*:}
  name=moe_e${E}_k${K}${SUFFIX}
  [[ -f runs/$name/DONE ]] && continue
  echo "== phase: $name  (E=$E, top-k=$K, gate_grad=$GATE_GRAD)"
  $PY -m moe.train --run "runs/$name" --resume --max-seconds "$MAX_SECONDS" \
      --init runs/dense/ckpt.pkl --upcycle --n-experts "$E" --top-k "$K" --gate-grad "$GATE_GRAD" \
      --steps "$STEPS" --lr 3e-4 --lr-final 3e-5 --warmup 50
  [[ -f runs/$name/DONE ]] || { echo "(paused: $name)"; exit 0; }
done
echo "SWEEP DONE — compare runs/moe_e*_k*/log.csv against runs/dense_cont and runs/moe"
