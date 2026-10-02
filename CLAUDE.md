# MoE-Upcycle — learning experiment: dense GPT → sparse Mixture-of-Experts

Context file for Claude Code sessions in this folder. Read fully before changing code.

## What this is

A from-scratch, dependency-light experiment (pure JAX + NumPy; **no Flax, no optax, no torch**)
that (1) trains a small dense char-level GPT on Shakespeare, (2) converts its dense MLPs into a
top-k routed Mixture-of-Experts by **sparse upcycling** (every expert = copy of the trained MLP,
plus a near-uniform router), and (3) keeps training, to show the MoE continues to learn — and
learns faster than the same dense model continued for the same steps on the same batches.

Purpose is *understanding MoE*, not SOTA. Everything is readable in `moe/model.py` (~200 lines)
and `moe/train.py`. Swati is working through this as part of ERA V5.

## Results so far (2026-09-29, 4-core CPU sandbox)

| run | steps | params total / active per token | val loss start → end |
|---|---|---|---|
| `runs/dense` (phase 1) | 0 → 2000 | 0.83M / 0.83M | 4.4292 → **1.6210** |
| `runs/dense_cont` (2a, control) | 2000 → 3500 | 0.83M / 0.83M | 1.6210 → **1.5390** |
| `runs/moe` (2b, E=4, top-2) | 2000 → 3500 | 2.41M / 1.36M | 1.6210 → **1.5186** |
| `runs/dense_cont_long` (2a-long, control) | 2000 → 7000 | 0.83M / 0.83M | 1.6210 → **1.4890** |
| `runs/moe_long` (2b-long, E=4, top-2) | 2000 → 7000 | 2.41M / 1.36M | 1.6210 → **1.4554** |

**Longer phase 2 (2026-09-29, Mac 12-core, `run_long.sh`, "next experiment #1" — answered):** the gap keeps
growing, no saturation by 5000 steps. MoE − dense at steps 3000/4000/5000/6000/7000 =
−0.013 / −0.018 / −0.023 / −0.029 / −0.034 nats/char (≈ −0.005 per 1000 steps in the second half).
Router load: layers 2–3 near-uniform; layer 0 lopsided (one expert ~0.46, another dipped to 0.06 at
step 6000, 0.14 at end) — motivates the aux_coef ablation (#3). Wall: dense 397 s, MoE 1560 s (dense-masked, E×).
Note: same schedule shape stretched over 5000 steps, so step-3500 values here are *higher* than the
1500-step runs (LR still high) — not a regression; compare only at equal step within the same experiment.

**E/k sweep (2026-09-30, `run_sweep.sh`, table via `python sweep_report.py`, plot `runs/sweep.png`; "next experiment #2" — answered).**
All arms: same dense parent, 1500 steps, phase-2 schedule, data-seed 1234; step-0 val = 1.6210 for every arm.

| run | E | k | params total / active | val @3500 | MoE − dense | aux @end | final loads |
|---|---|---|---|---|---|---|---|
| `dense_cont` | – | – | 0.83M / 0.83M | 1.5390 | – | – | – |
| `moe_e2_k1` | 2 | 1 | 1.36M / 0.83M | 1.5353 | −0.0037 | 1.000 | 0.50/0.50 all layers |
| `moe_e4_k1` | 4 | 1 | 2.41M / 0.83M | 1.5286 | −0.0104 | 1.000 | ~0.25 each, all layers |
| `moe_e8_k1` | 8 | 1 | 4.52M / 0.83M | 1.5330 | −0.0060 | 1.000 | ~0.125 each, all layers |
| `moe_e2_k2` | 2 | 2 | 1.36M / 1.36M | 1.5306 | −0.0084 | 1.000 | 0.50/0.50 (trivially: k=E) |
| `moe` | 4 | 2 | 2.41M / 1.36M | 1.5186 | −0.0204 | 1.086 | 0.10–0.45 |
| `moe_e8_k2` | 8 | 2 | 4.52M / 1.36M | **1.5117** | **−0.0273** | 1.294 | very skewed: one expert 0.48–0.50 in layers 0–1, one at 0.00 |

Findings:
- Every arm beats the control; top-2 arms beat top-1 arms; more experts helps at top-2 (E=8 best) but
  not monotonically at top-1 (E=4 > E=8 > E=2).
- **Top-1 + renormalised gates = frozen router.** With k=1, `gates_k = topv/topv.sum() ≡ 1`, so
  d(gate)/d(router) = 0 exactly and the router gets *no* LM-loss gradient — only the aux loss, which drives
  it to uniform. Evidence: all top-1 arms end with aux = 1.000 and perfectly uniform loads, and router
  RMS barely moves (1.1–1.7e-3 vs init 1e-3), whereas top-2 routers grow ~10× (1e-2 to 2e-2). So the top-1
  arms are effectively *fixed pseudo-random routing* (hash-routing-like), which still helps a little
  (−0.004 to −0.010) but is not learned routing. `moe_e4_k1` (same active params as dense) beating
  `dense_cont` by 0.010 is therefore a "more total params, random split" effect, not a routing effect.
  Fix candidates (both keep exact function-preservation at conversion): (a) `gates_k = topv / stop_gradient(topv.sum())`
  so gradient flows through the selected prob; (b) Switch-style un-renormalised gate `probs` for k=1 plus
  a compensating ×E scale at upcycle time. Needs Swati's call before changing `moe_ffn` — see next experiments.
- `moe_e2_k2` is not sparse (k=E): both experts always on, so it is a gated 2×-wide MLP; a useful
  "just make the MLP wider" reference point alongside the planned d_ff=1024 control (#6).
- E=8 top-2 is the best arm but the *least* balanced (aux 1.29, one expert idle in layer 0) — aux_coef=0.01
  is too weak at E=8; motivates ablation #3 with E=8.
- Wall time per 1500 steps (dense-masked): E=2 ~225 s, E=4 ~460 s, E=8 ~990 s (≈ linear in E, as expected).

**2b: `gate_grad=stopgrad` A/B (2026-09-30, `GRID="4:2 4:1 2:1 8:1" GATE_GRAD=stopgrad SUFFIX=_sg bash run_sweep.sh`) — the fix makes the top-1 router learn, and that learning is HARMFUL here.**
Implementation: `cfg["gate_grad"]` ∈ {`renorm` (default, original), `stopgrad`}; `--gate-grad` CLI, stored in cfg at upcycle;
old checkpoints lack the key and behave exactly as before. Tests: `tests/test_gating.py` (5/5). Forward values identical in both modes.

| run | E | k | gate | val @3500 | vs dense | aux @end | loads |
|---|---|---|---|---|---|---|---|
| `moe` | 4 | 2 | renorm | 1.5186 | −0.0204 | 1.086 | 0.10–0.45 |
| `moe_e4_k2_sg` | 4 | 2 | stopgrad | 1.5332 | −0.0058 | 1.121 | layer 3: 0.03/0.35/0.16/0.46 |
| `moe_e2_k1` / `_sg` | 2 | 1 | renorm / stopgrad | 1.5353 / 1.5385 | −0.0037 / −0.0005 | 1.000 / 1.088 | uniform / up to 0.76 |
| `moe_e4_k1` / `_sg` | 4 | 1 | renorm / stopgrad | 1.5286 / **1.5491** | −0.0104 / **+0.0101** | 1.000 / 1.227 | uniform / layer 3: 0.75 on one expert |
| `moe_e8_k1` / `_sg` | 8 | 1 | renorm / stopgrad | 1.5330 / **1.5551** | −0.0060 / **+0.0161** | 1.000 / 1.439 | uniform / layer 3: 0.71 on one, four at ≤0.01 |

Why: with the denominator stop-gradiented, ∂L/∂p_sel = ⟨∂L/∂y, y_e⟩ / Σp no longer cancels across the selected
set. For k=1 it is a pure "raise confidence in whichever expert was picked if its output currently helps" signal.
At the upcycle point all experts are identical, so this carries no information about *which* expert is better —
it just rewards the incumbent → rich-get-richer → collapse (0.71–0.75 loads), aux loss climbs to 1.2–1.4 fighting
it, and the val loss first *rises* above 1.6210 (E=4: 1.6452 at step 2100) before recovering to worse-than-dense.
For k=2 the same common-mode bias is added on top of the useful relative gradient → 0.015 worse than renorm.
Conclusion: keep `renorm` as the default and the reference; top-1 upcycling needs a different gate
(Switch-style raw prob with ×E rescale at conversion, or noisy/tie-broken routing) — Swati's call, see #2c.

**#3 Router ablations on E=8 top-2 (2026-09-30, `run_ablation.sh`; diagnostics `python router_analysis.py runs/moe_e8_k2*`).**
Baseline `moe_e8_k2` (aux 0.01, router_std 1e-3). One knob per arm; same parent/steps/schedule/batches. Val = 20 fixed val batches.

| run | aux_coef | router_std | val @3500 | vs baseline | vs dense (1.5390) | worst max-load | dead experts (<1%, 4 layers×8) |
|---|---|---|---|---|---|---|---|
| `moe_e8_k2` | 0.01 | 1e-3 | 1.5117 | – | −0.027 | 0.50 | 2 |
| `moe_e8_k2_aux0` | **0** | 1e-3 | 1.5251 | +0.013 | −0.014 | 0.50 | **20** |
| `moe_e8_k2_aux0p1` | **0.1** | 1e-3 | 1.5037 | −0.008 | −0.035 | 0.15 | 0 |
| `moe_e8_k2_std0` | 0.01 | **0** | 1.5369 | +0.025 | −0.002 | 0.14 | 0 |
| `moe_e8_k2_std0p01` | 0.01 | **1e-2** | **1.5007** | **−0.011** | **−0.038** | 0.41 | 0 |

Findings:
- **aux_coef=0 → textbook collapse.** In layers 0–2 exactly two experts take 100% of assignments (0.50/0.50),
  i.e. with top-2 those layers are a *dense* MLP again; 20 of 32 expert slots are dead. Still beats dense
  (−0.014) because layer 3 keeps 4 live experts, but loses 0.013 vs the balanced baseline.
- **aux_coef=0.1 → perfectly balanced (max-load 0.15 ≈ 1/8) and better** (1.5037). At E=8 the default 0.01 is too
  weak; the sweep's "E=8 top-2 is best but least balanced" was leaving loss on the table.
- **router_std=0 → experts never separate.** With exact ties, `top_k` picks (0,1) for every token; the aux loss then
  rotates selection through (2,3), (4,5), (6,7), but each *pair* is always selected together with gates 0.5/0.5, so
  both members get bit-identical gradients forever. Checked on the checkpoint: experts (0,1),(2,3),(4,5),(6,7) are
  exactly identical in every layer (max |Δw1| = 0.0, router columns identical), non-twin pairs differ by ≥0.017.
  The model is effectively E=4 top-1 with aux-driven (uninformative) routing → val 1.5369 ≈ dense. Router RMS stays
  ~1e-3 (only the aux loss moves it). *Symmetry breaking at upcycle is mandatory; the aux loss alone cannot do it.*
- **router_std=1e-2 → best run in the project so far** (1.5007). Larger initial router noise = faster
  symmetry breaking / earlier specialisation; step-0 val still 1.6210 (renormalised gates make the noise irrelevant
  at conversion). Loads less even than aux 0.1 (worst 0.41) — so **combining std 1e-2 with aux 0.1 is the obvious next arm**.
- H(P)/ln E (mean router prob entropy) is ~1.0 even when assignments are skewed — the *mean* softmax is near-uniform
  while top-k picks are not. Use max-load / dead count for collapse, not P-entropy.

**#3b (2026-09-30): combining the winners, and is "aux 0.01 too weak" E=8-specific?**

| run | E | aux_coef | router_std | val @3500 | vs dense | worst max-load |
|---|---|---|---|---|---|---|
| `moe_e8_k2_aux0p1_std0p01` | 8 | 0.1 | 1e-2 | **1.4976** | **−0.0414** | 0.15 |
| `moe_e8_k2_std0p01` | 8 | 0.01 | 1e-2 | 1.5007 | −0.0383 | 0.41 |
| `moe_e8_k2_aux0p1` | 8 | 0.1 | 1e-3 | 1.5037 | −0.0353 | 0.15 |
| `moe` | 4 | 0.01 | 1e-3 | 1.5186 | −0.0204 | 0.44 |
| `moe_e4_k2_aux0p1` | 4 | 0.1 | 1e-3 | 1.5176 | −0.0214 | 0.27 |

- The two knobs are additive: aux 0.1 + std 1e-2 gives the **best run in the project (1.4976)**, with perfectly
  balanced loads. Gains: +0.003 over std-only, +0.006 over aux-only, +0.014 over the E=8 baseline, 2× the E=4 gap.
- At E=4, aux 0.1 vs 0.01 is a wash (1.5176 vs 1.5186; loads even out 0.44 → 0.27 but loss barely moves). So the
  balancing-strength sensitivity is E=8-specific: with only 4 experts and top-2, the default 0.01 already keeps
  every expert alive; with 8 it does not. Rule of thumb from this toy: aux_coef needs to scale up with E.
- **Recommended MoE settings (docs only; train.py defaults unchanged for reproducibility of `runs/moe`):**
  E=8, top-2, `--aux-coef 0.1 --router-std 1e-2`, renorm gating. Use these for #4–#7 unless the experiment is about them.

**Error bars (2026-09-30, `run_seeds.sh` / `seeds_report.py`):** 3 paired seeds (data-seed 1234, 1, 2) of dense control vs
recommended MoE. Dense 1.5395 ± 0.0013, MoE 1.4995 ± 0.0017, **paired gap −0.0400 ± 0.0015** (all three negative:
−0.0414 / −0.0384 / −0.0401). Rule for reading single-seed numbers in this file: ±0.0015 nats/char. Differences
< ~0.005 between arms (e.g. the top three ablation arms) are not rankings.

**Wrap-up status:** core experiment closed 2026-09-30 — `RESULTS.md` is the narrative, `runs/headline.png` the figure,
`verify.py` (4 checks, 19 upcycled runs) the machine-checked claims. Remaining queue items (#4–#7, #2c) are extensions.

Verified (`python verify.py`):
- Upcycling is function-preserving: max |dense logits − upcycled MoE logits| = 3e-5 (fp noise);
  the MoE's step-0 val loss equals the dense parent's exactly (1.6210).
- Experts diverge during training: pairwise relative distance of expert `w1` goes 0.00 → 0.12–0.24
  (more in later layers).
- MoE beats the control by 0.020 nats/char at step 3500 and the gap is still widening.
- Router load stays reasonably balanced with aux_coef=0.01 (per-layer loads ~0.10–0.45), aux loss ~1.09–1.13 (min 1.0).

Plot: `runs/loss_curves.png`. Raw numbers: `runs/*/log.csv`. Samples: `runs/*/sample.txt`.

## Layout

```
data/input.txt            5.06M chars, 42 Shakespeare plays (Gutenberg text via PyPI `shakespeare` pkg), ASCII-only, vocab 84
moe/model.py              init_dense, upcycle_dense_to_moe, forward (dense_mlp | moe_ffn), loss_fn, generate
moe/train.py              CLI trainer; hand-rolled AdamW; resumable full-state checkpoints
moe/data.py               char dataset; last 5% = validation; deterministic val windows
run_experiment.sh         dense → dense_cont + moe; idempotent/resumable via runs/<name>/DONE
run_long.sh               next-experiment #1: dense_cont_long + moe_long (5000-step phase 2), same pattern
run_sweep.sh              next-experiment #2: runs/moe_e{E}_k{K} for E∈{2,4,8}, k∈{1,2} (4:2 = runs/moe); GRID/STEPS env overrides
sweep_report.py           markdown table of the sweep vs dense_cont (+ step-0 preservation check) and runs/sweep.png
tests/test_gating.py      moe_ffn gating tests: function preservation (both gate modes), top-1 renorm has no router grad, stopgrad does
run_ablation.sh           next-experiment #3/#3b: E=8 top-2 with aux_coef∈{0,0.1}, router_std∈{0,1e-2}, both combined, + E=4 aux 0.1 → runs/moe_e8_k2_{aux0,aux0p1,std0,std0p01,aux0p1_std0p01}, runs/moe_e4_k2_aux0p1
router_analysis.py        per-layer loads, max-load, dead experts, router entropy/RMS + val on fixed val batches, for any MoE ckpts; markdown summary
run_seeds.sh              wrap-up: 2 extra PAIRED seeds (data-seed s for both arms) of dense control vs recommended E=8 MoE → runs/{dense_cont,moe_e8_rec}_s{1,2}
seeds_report.py           per-pair gap + mean ± std over the 3 pairs (orig 1234 + s1 + s2)
RESULTS.md                narrative write-up of all findings (for the ERA session); headline figure runs/headline.png (plot.py)
requirements.txt/.lock    top-level deps / full uv-compiled pin set; .venv = uv venv, CPython 3.13, jax 0.11.2
plot.py                   loss curves (full + phase-2 zoom); picks up *_long runs when present, prints MoE−dense gap
verify.py                 sanity checks: [1] preservation E=4 & E=8, [2] expert drift, [3] short/long/recommended gaps, [4] all upcycled runs start at parent loss
runs/<name>/              ckpt.pkl (params+opt+rng+phase_step), log.csv, config.json, DONE, sample.txt
runs/*.out                stdout of the batch launchers — the ONLY place per-layer expert loads during training are recorded; keep
(runs_old*/ moved to ~/.Trash/MoE-Upcycle_runs_old* on 2026-09-30)
```

## Model / training config (defaults in train.py)

- GPT: 4 layers, d_model 128, 4 heads, d_ff 512, block 128, pre-LN, GELU, no dropout, untied head.
- MoE block: `n_experts=4`, `top_k=2`, softmax router, top-k gates renormalised to sum to 1,
  Switch-style load-balancing aux loss (E·Σ f_e·P_e, mean over MoE layers), `aux_coef=0.01`.
  **Recommended for new experiments (from ablation #3/#3b, 2026-09-30):** `--n-experts 8 --top-k 2 --aux-coef 0.1 --router-std 1e-2`
  → val 1.4976 @3500 vs 1.5186 for the E=4 defaults and 1.5390 dense. CLI defaults intentionally left as-is so `runs/moe` reproduces.
- Router init std 1e-3 (`--router-std`; tiny, non-zero → breaks ties so experts can diverge. **Must be > 0**: with 0,
  experts stay bit-identical in pairs forever — ablation #3. 1e-2 trained better than 1e-3 at E=8.)
- AdamW β=(0.9,0.95), wd 0.1 on ≥2-D params, grad-clip 1.0, warmup + cosine.
  Phase 1: lr 1e-3 → 3e-4. Phase 2 (both arms): lr 3e-4 → 3e-5, 50 warmup, `--data-seed 1234` so both arms see identical batches.
- Batch 32 × 128 tokens. Dense ≈ 0.25 s/step, MoE ≈ 1 s/step on 4 CPU cores.

## Design decisions — keep unless Swati says otherwise

- **Pure JAX, explicit pytrees, hand-rolled optimizer.** The point is that nothing is hidden. Don't
  introduce Flax/optax/torch "for convenience". A separate PyTorch port is an *optional* task (below).
- **Upcycle must be exactly function-preserving.** `upcycle_dense_to_moe` copies the MLP into every
  expert; with renormalised top-k gates the output is identical to the dense model. `verify.py` check [1]
  must stay at ~1e-5. Any router/gating change must keep this property at conversion time.
  *Known caveat:* the default renormalisation makes the top-1 gate identically 1, so with k=1 the router
  receives no LM gradient (sweep finding, 2026-09-30). Top-1 `renorm` results are valid as "fixed random routing" only.
  `--gate-grad stopgrad` is the obvious fix and is implemented, but it *hurts* (collapse; see Results 2b) — don't
  make it the default. Any new gating must pass `tests/test_gating.py` (function preservation for both modes).
- **Control arm is mandatory.** Any MoE claim is relative to `dense_cont` trained from the same
  checkpoint with the same `--data-seed`. Don't compare against phase-1 alone.
- **MoE is computed "dense-masked"** (all experts on all tokens, zero gates for unselected). This is
  mathematically identical to sparse dispatch but E× the FLOPs. Fine for the toy; note it whenever
  quoting speed. Real dispatch is a follow-up task, not a bug.
- **Resumability is a hard requirement.** `ckpt.pkl` carries params, AdamW moments, NumPy RNG state,
  and `phase_step`. `--resume` must continue the LR schedule and batch order exactly. Keep atomic
  writes (`.tmp` + `os.replace`). Any trainer change: run a short `--max-seconds` pause/resume and
  confirm log.csv continues smoothly.
- **Never delete `runs/` results**; make a new run name (`runs/moe_e8`, …) instead.

## Commands

```bash
cd ~/Claude/Projects/Personal/MoE-Upcycle          # adjust if moved
# env (already set up on the Mac, 2026-09-29): uv venv, CPython 3.13, pinned in requirements.lock.
# A Claude Code hook blocks any `pip install` string, and `uv pip sync` from the 4-line requirements.txt
# skips transitive deps (ml_dtypes, scipy, opt_einsum) — always sync from the compiled lock:
uv venv .venv --python 3.13
uv pip compile requirements.txt -o requirements.lock --python .venv/bin/python
uv pip sync --python .venv/bin/python requirements.lock
source .venv/bin/activate                           # or prefix commands with .venv/bin/python

bash run_experiment.sh                              # reproduces the 1500-step experiment (idempotent; ~35 min on 4 cores, ~10 min on the Mac)
MAX_SECONDS=600 bash run_experiment.sh              # time-boxed chunks; rerun to continue
bash run_long.sh                                    # 5000-step phase 2 (done: runs/dense_cont_long, runs/moe_long; ~33 min on the Mac)
bash run_sweep.sh && python3 sweep_report.py        # E/k sweep (done); GRID="4:2" GATE_GRAD=stopgrad SUFFIX=_sg for the gating A/B
bash run_ablation.sh && python3 router_analysis.py runs/moe_e8_k2*   # router ablations (done)
python3 verify.py && python3 plot.py

# individual phases
python3 -m moe.train --run runs/dense --steps 2000 --lr 1e-3 --lr-final 3e-4 --warmup 100 --resume
python3 -m moe.train --run runs/dense_cont --init runs/dense/ckpt.pkl --steps 1500 --lr 3e-4 --lr-final 3e-5 --warmup 50 --resume
python3 -m moe.train --run runs/moe --init runs/dense/ckpt.pkl --upcycle --n-experts 4 --top-k 2 \
        --steps 1500 --lr 3e-4 --lr-final 3e-5 --warmup 50 --resume

# extend an existing run by N more steps: bump --steps (schedule is recomputed over the new total,
# so prefer a NEW run name with --init for clean curves)
python3 -m moe.train --run runs/moe_ext --init runs/moe/ckpt.pkl --steps 3000 --lr 1e-4 --lr-final 1e-5
```

Only PyPI was reachable when this was built (no HF / GitHub); the corpus was pulled from the
`shakespeare==0.6` sdist (`shksprdata/texts/*_gut.txt`). On the Mac, HF `tiny_shakespeare` or any
text file dropped into `data/input.txt` works — vocab must match a checkpoint you `--init` from.

## Next experiments (priority order)

1. ~~**Longer phase 2** (5000 steps each arm)~~ **DONE 2026-09-29** (`run_long.sh`, see Results): gap keeps
   growing to −0.034 at step 7000, no saturation. Follow-up if wanted: 10k+ steps, or repeat with 2–3 seeds
   to put error bars on the gap.
2. ~~**Sweep E and k**~~ **DONE 2026-09-30** (`run_sweep.sh`, see Results): all arms beat dense; E=8 top-2 best
   (−0.027); E=4 top-1 beats dense by 0.010 at equal active params — BUT top-1 routers are frozen (see finding).
   ~~**2b: fix the top-1 router gradient via `topv / stop_gradient(sum)`**~~ **DONE 2026-09-30 — negative result**
   (see Results): router learns but collapses; every stopgrad arm is worse than its renorm twin, top-1 arms worse
   than dense. `--gate-grad stopgrad` stays available as a documented ablation; default remains `renorm`.
   **2c (needs Swati's decision): a top-1 gate that is both function-preserving and informative.** Options:
   (a) Switch-style gate = p_sel, with expert `w2`/`b2` scaled ×E at upcycle and `router_std=0` so p_sel = 1/E exactly
   at conversion (needs a tie-break: tiny token-dependent noise on router logits, or `jax.lax.top_k` index order);
   (b) skip top-1 entirely — top-2 renorm is where learned routing demonstrably works — and go to #3 with E=8 top-2.
   Recommendation: (b) first, (a) only if the top-1-at-equal-active-params story matters for the write-up.
3. ~~**Router ablations**: aux_coef ∈ {0, 0.01, 0.1}; router_std ∈ {0, 1e-3, 1e-2}~~ **DONE 2026-09-30** (see Results):
   aux=0 collapses to 2 live experts/layer; aux=0.1 balanced and better; std=0 leaves 4 bit-identical expert pairs
   (≈ dense); std=1e-2 is the best run so far (1.5007). ~~**3b:** combine the two winners~~ **DONE 2026-09-30**:
   `moe_e8_k2_aux0p1_std0p01` = **1.4976**, additive gains, balanced; aux 0.1 at E=4 is a wash → sensitivity is
   E-specific. Recommended settings recorded in the config section. Open follow-up (cheap): aux_coef 0.3 / 1.0 at E=8
   to find where over-balancing starts to hurt.
4. **Expert specialisation analysis**: for the trained MoE, dump which characters/contexts route to
   which expert (e.g. by preceding char class: letter/space/punct/newline, or by speaker-name lines).
   Good notebook material.
5. **Real sparse dispatch**: implement gather→expert→scatter with capacity factor; check it matches the
   dense-masked output bit-for-bit (up to fp) and measure the speedup.
6. **Fair-FLOPs control**: train a dense model with d_ff=1024 (same active params as top-2/E=4) as a
   second control — MoE vs "just make the MLP wider".
7. **Upcycle from a *longer*-trained dense model** (e.g. 10k steps): the Sparse Upcycling paper's
   claim is that upcycling beats training MoE from scratch at equal total compute — test the
   from-scratch MoE arm too (`--n-experts` on a fresh init: needs a small `init_moe` helper).
8. Optional: PyTorch port in `torch_port/` mirroring `model.py` 1:1 (keep JAX as the reference).

## Guardrails for Claude Code

- After any change to `model.py` or `train.py`: run `python3 tests/test_gating.py`, `python3 verify.py` (needs existing `runs/`),
  and a 30-step smoke run: `python3 -m moe.train --run /tmp/smoke --steps 30 --eval-every 10 --eval-batches 2`.
- Keep `log.csv` schema stable (`step,train_loss,val_loss,aux_loss,lr,grad_norm,sec`); plot.py and verify.py read it.
- Report losses in nats/char; always state total *and* active params alongside any MoE number.
- Don't quote MoE step-time as "cost" without noting the dense-masked implementation.
