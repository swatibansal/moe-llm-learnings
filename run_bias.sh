#!/usr/bin/env bash
# Aux-loss-free balancing (DeepSeek-V3 style selection bias, --bias-gamma) on E=8 top-2, router_std 1e-2.
# Comparators already in runs/: moe_e8_k2_aux0p1_std0p01 (aux 0.1 = best, 1.4976), moe_e8_k2_std0p01 (aux 0.01, 1.5007),
# moe_e8_k2_std0 (aux 0.01, std 0: experts stay in identical pairs, 1.5369).
# Arms (one idea each):
#   bias0p001            aux 0,   gamma 1e-3            bias alone (paper default)
#   aux0_std0p01         aux 0,   gamma 0               exact no-balancing control at this router_std (collapse expected)
#   bias0p01             aux 0,   gamma 1e-2            10x step: does a coarse bias oscillate / hurt?
#   bias0p001_aux0p1     aux 0.1, gamma 1e-3            both (DeepSeek keeps a small aux term too)
#   bias0p001_std0       aux 0,   gamma 1e-3, std 0     prediction: bias breaks the exact-tie symmetry that aux could not
# Same parent/steps/schedule/--data-seed as everything else. ~17 min/arm on the Mac. Idempotent + resumable.
set -euo pipefail
cd "$(dirname "$0")"
[[ -x .venv/bin/python ]] && PY=.venv/bin/python || PY=python3
STEPS=${STEPS:-1500}; E=${E:-8}; K=${K:-2}
MAX_SECONDS=${MAX_SECONDS:-0}
export PYTHONUNBUFFERED=1
[[ -f runs/dense/DONE ]] || { echo "runs/dense not finished; run run_experiment.sh first"; exit 1; }

phase() {  # name aux_coef bias_gamma router_std
  local name=$1 aux=$2 gamma=$3 std=$4
  [[ -f runs/$name/DONE ]] && return 0
  echo "== phase: $name  (E=$E, top-k=$K, aux_coef=$aux, bias_gamma=$gamma, router_std=$std)"
  $PY -m moe.train --run "runs/$name" --resume --max-seconds "$MAX_SECONDS" \
      --init runs/dense/ckpt.pkl --upcycle --n-experts "$E" --top-k "$K" \
      --aux-coef "$aux" --bias-gamma "$gamma" --router-std "$std" \
      --steps "$STEPS" --lr 3e-4 --lr-final 3e-5 --warmup 50
  [[ -f runs/$name/DONE ]] || { echo "(paused: $name)"; exit 0; }
}

phase moe_e${E}_k${K}_bias0p001          0    1e-3  1e-2
phase moe_e${E}_k${K}_aux0_std0p01       0    0     1e-2
phase moe_e${E}_k${K}_bias0p01           0    1e-2  1e-2
phase moe_e${E}_k${K}_bias0p001_aux0p1   0.1  1e-3  1e-2
phase moe_e${E}_k${K}_bias0p001_std0     0    1e-3  0
echo "BIAS DONE — python router_analysis.py runs/moe_e${E}_k${K}_bias* runs/moe_e${E}_k${K}_aux0_std0p01 runs/moe_e${E}_k${K}_aux0p1_std0p01"
