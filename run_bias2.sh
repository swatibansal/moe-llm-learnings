#!/usr/bin/env bash
# Bounded-score selection bias (--bias-on probs), after run_bias.sh showed that biasing unbounded LOGITS cannot
# keep up with a confident router (|bias| hit its 1.5 ceiling, layers 0-1 still ~50% on one expert).
# Softmax probs are in [0,1], like the paper's sigmoid affinities, so a bias of order 1 can always flip a choice.
# E=8 top-2, router_std 1e-2, same parent/steps/schedule/--data-seed. ~17 min/arm. Idempotent + resumable.
#   pbias0p001           aux 0,   gamma 1e-3  paper-faithful: bias alone, bounded scores
#   pbias0p01            aux 0,   gamma 1e-2  10x step on bounded scores
#   pbias0p001_aux0p1    aux 0.1, gamma 1e-3  both
set -euo pipefail
cd "$(dirname "$0")"
[[ -x .venv/bin/python ]] && PY=.venv/bin/python || PY=python3
STEPS=${STEPS:-1500}; E=${E:-8}; K=${K:-2}
MAX_SECONDS=${MAX_SECONDS:-0}
export PYTHONUNBUFFERED=1
[[ -f runs/dense/DONE ]] || { echo "runs/dense not finished; run run_experiment.sh first"; exit 1; }

phase() {  # name aux_coef bias_gamma
  local name=$1 aux=$2 gamma=$3
  [[ -f runs/$name/DONE ]] && return 0
  echo "== phase: $name  (E=$E, top-k=$K, aux_coef=$aux, bias_gamma=$gamma, bias_on=probs, router_std=1e-2)"
  $PY -m moe.train --run "runs/$name" --resume --max-seconds "$MAX_SECONDS" \
      --init runs/dense/ckpt.pkl --upcycle --n-experts "$E" --top-k "$K" \
      --aux-coef "$aux" --bias-gamma "$gamma" --bias-on probs --router-std 1e-2 \
      --steps "$STEPS" --lr 3e-4 --lr-final 3e-5 --warmup 50
  [[ -f runs/$name/DONE ]] || { echo "(paused: $name)"; exit 0; }
}

phase moe_e${E}_k${K}_pbias0p001         0    1e-3
phase moe_e${E}_k${K}_pbias0p01          0    1e-2
phase moe_e${E}_k${K}_pbias0p001_aux0p1  0.1  1e-3
echo "BIAS2 DONE — python router_analysis.py runs/moe_e${E}_k${K}_*bias* runs/moe_e${E}_k${K}_aux0_std0p01 runs/moe_e${E}_k${K}_aux0p1_std0p01"
