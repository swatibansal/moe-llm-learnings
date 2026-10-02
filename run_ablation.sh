#!/usr/bin/env bash
# Next experiment #3 from CLAUDE.md: router ablations on the best sweep config, E=8 top-2 (renorm gating).
# Baseline is runs/moe_e8_k2 (aux_coef=0.01, router_std=1e-3). Each arm changes ONE knob:
#   aux_coef   in {0, 0.1}      -> runs/moe_e8_k2_aux0, runs/moe_e8_k2_aux0p1   (collapse when aux=0?)
#   router_std in {0, 1e-2}     -> runs/moe_e8_k2_std0, runs/moe_e8_k2_std0p01  (std=0: exact ties; can aux alone break symmetry?)
# Same dense parent, 1500 steps, phase-2 schedule, --data-seed 1234. ~16 min per arm on the Mac (E=8 dense-masked).
# Idempotent + resumable:  MAX_SECONDS=600 bash run_ablation.sh ; rerun to continue.
set -euo pipefail
cd "$(dirname "$0")"
[[ -x .venv/bin/python ]] && PY=.venv/bin/python || PY=python3
STEPS=${STEPS:-1500}; E=${E:-8}; K=${K:-2}
MAX_SECONDS=${MAX_SECONDS:-0}
export PYTHONUNBUFFERED=1
[[ -f runs/dense/DONE ]] || { echo "runs/dense not finished; run run_experiment.sh first"; exit 1; }

phase() {  # name aux_coef router_std [n_experts]
  local name=$1 aux=$2 std=$3 e=${4:-$E}
  [[ -f runs/$name/DONE ]] && return 0
  echo "== phase: $name  (E=$e, top-k=$K, aux_coef=$aux, router_std=$std)"
  $PY -m moe.train --run "runs/$name" --resume --max-seconds "$MAX_SECONDS" \
      --init runs/dense/ckpt.pkl --upcycle --n-experts "$e" --top-k "$K" \
      --aux-coef "$aux" --router-std "$std" \
      --steps "$STEPS" --lr 3e-4 --lr-final 3e-5 --warmup 50
  [[ -f runs/$name/DONE ]] || { echo "(paused: $name)"; exit 0; }
}

phase moe_e${E}_k${K}_aux0    0     1e-3
phase moe_e${E}_k${K}_aux0p1  0.1   1e-3
phase moe_e${E}_k${K}_std0    0.01  0
phase moe_e${E}_k${K}_std0p01 0.01  1e-2
# 3b: combine the two winners; and check whether "aux 0.01 is too weak" is E=8-specific
phase moe_e${E}_k${K}_aux0p1_std0p01 0.1 1e-2
phase moe_e4_k${K}_aux0p1            0.1 1e-3  4
echo "ABLATION DONE — python router_analysis.py runs/moe_e${E}_k${K}* runs/moe runs/moe_e4_k${K}_aux0p1"
