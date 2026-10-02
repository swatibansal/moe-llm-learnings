#!/usr/bin/env bash
# Wrap-up item: error bars. Two extra PAIRED seeds of (dense control, recommended MoE) from the same dense parent.
# Pair s uses --data-seed s for BOTH arms (identical batches within a pair) and --seed s for the MoE router init.
# Seed-0 pair already exists: runs/dense_cont (data-seed 1234) vs runs/moe_e8_k2_aux0p1_std0p01 (data-seed 1234, seed 0).
# Recommended MoE = E=8 top-2, aux 0.1, router_std 1e-2 (ablation #3b). 1500 steps, phase-2 schedule.
# ~2 min per dense arm, ~17 min per MoE arm on the Mac. Idempotent + resumable (MAX_SECONDS=600 ...).
set -euo pipefail
cd "$(dirname "$0")"
[[ -x .venv/bin/python ]] && PY=.venv/bin/python || PY=python3
STEPS=${STEPS:-1500}; SEEDS=${SEEDS:-"1 2"}
MAX_SECONDS=${MAX_SECONDS:-0}
export PYTHONUNBUFFERED=1
[[ -f runs/dense/DONE ]] || { echo "runs/dense not finished; run run_experiment.sh first"; exit 1; }

phase() {  # name, then train args
  local name=$1; shift
  [[ -f runs/$name/DONE ]] && return 0
  echo "== phase: $name"
  $PY -m moe.train --run "runs/$name" --resume --max-seconds "$MAX_SECONDS" --init runs/dense/ckpt.pkl \
      --steps "$STEPS" --lr 3e-4 --lr-final 3e-5 --warmup 50 "$@"
  [[ -f runs/$name/DONE ]] || { echo "(paused: $name)"; exit 0; }
}

for s in $SEEDS; do
  phase dense_cont_s$s   --data-seed "$s"
  phase moe_e8_rec_s$s   --data-seed "$s" --seed "$s" --upcycle --n-experts 8 --top-k 2 --aux-coef 0.1 --router-std 1e-2
done
echo "SEEDS DONE — python seeds_report.py"
