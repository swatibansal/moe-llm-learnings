"""Sanity checks on the experiment.
1. Upcycling is function-preserving: MoE(dense ckpt) logits == dense logits on a batch
   (E=4 default and E=8 with the recommended router_std=1e-2).
2. After training, experts have actually diverged from each other (specialisation).
3. Val loss keeps falling after conversion, and beats the dense-continued control
   (original 1500-step pair; 5000-step pair with a WIDENING gap; recommended E=8 config beats E=4).
4. Every upcycled run in runs/ logged the dense parent's val loss at step 0 (global preservation check).
Checks whose runs are missing are reported as SKIP, not failed.
"""
import csv, glob, json, os, numpy as np, jax, jax.numpy as jnp
from moe import model as M
from moe.train import load_ckpt
from moe.data import CharData

data = CharData("data/input.txt", block=128)
x, y = next(data.val_batches(16, 1))
dense = load_ckpt("runs/dense/ckpt.pkl")
ld, _, _ = M.forward(dense["params"], x, dense["cfg"])
for E, std in [(4, 1e-3), (8, 1e-2)]:
    moe0 = M.upcycle_dense_to_moe(dense["params"], E, jax.random.PRNGKey(0), router_std=std)
    lm, _, _ = M.forward(moe0, x, {**dense["cfg"], "n_experts": E, "top_k": 2})
    err = float(jnp.abs(ld - lm).max())
    print(f"[1] E={E} router_std={std:g}: max |dense logits - upcycled MoE logits| = {err:.2e}  (should be ~1e-5, fp noise)")
    assert err < 1e-3, "upcycle is not function-preserving"

moe = load_ckpt("runs/moe/ckpt.pkl")["params"]
for i, blk in enumerate(moe["blocks"]):
    w1 = blk["moe"]["w1"]                         # (E,D,H)
    diffs = [float(jnp.linalg.norm(w1[a] - w1[b]) / jnp.linalg.norm(w1[a])) for a in range(4) for b in range(a + 1, 4)]
    print(f"[2] layer {i}: expert w1 pairwise relative distance  mean {np.mean(diffs):.3f}  (0.000 at conversion)")

def last(run):
    rows = [r for r in csv.DictReader(open(f"runs/{run}/log.csv")) if r["val_loss"]]
    return float(rows[0]["val_loss"]), float(rows[-1]["val_loss"])
d0, d1 = last("dense_cont"); m0, m1 = last("moe")
print(f"[3] dense_cont  val {d0:.4f} -> {d1:.4f}   (Δ {d1-d0:+.4f})")
print(f"[3] moe         val {m0:.4f} -> {m1:.4f}   (Δ {m1-m0:+.4f})   MoE − dense at end: {m1-d1:+.4f}")
assert m1 < m0 and m1 < d1, "MoE should keep improving and beat the control"

if os.path.exists("runs/moe_long/log.csv") and os.path.exists("runs/dense_cont_long/log.csv"):
    D0, D1 = last("dense_cont_long"); L0, L1 = last("moe_long")
    gap_long, gap_short = L1 - D1, m1 - d1
    print(f"[3] 5000-step pair: dense {D1:.4f}, MoE {L1:.4f}, gap {gap_long:+.4f} @7000 vs {gap_short:+.4f} @3500 (should widen)")
    assert L1 < D1 and gap_long < gap_short, "long-run MoE gap should be negative and wider than the 1500-step gap"
else:
    print("[3] 5000-step pair: SKIP (run_long.sh not run)")

REC = "moe_e8_k2_aux0p1_std0p01"
if os.path.exists(f"runs/{REC}/log.csv"):
    r0, r1 = last(REC)
    print(f"[3] recommended E=8 (aux 0.1, std 1e-2): val {r0:.4f} -> {r1:.4f}   vs E=4 default {m1:.4f}, dense {d1:.4f}")
    assert r1 < m1 < d1, "recommended config should beat E=4 default which beats dense"
else:
    print(f"[3] recommended E=8: SKIP (runs/{REC} missing)")

# [4] every upcycled run starts at the dense parent's loss
parent_end = float([r for r in csv.DictReader(open("runs/dense/log.csv")) if r["val_loss"]][-1]["val_loss"])
bad, n = [], 0
for cfgp in sorted(glob.glob("runs/*/config.json")):
    run = os.path.dirname(cfgp)
    meta = json.load(open(cfgp))
    if not meta["args"].get("upcycle") or not os.path.exists(f"{run}/log.csv"):
        continue
    rows = [r for r in csv.DictReader(open(f"{run}/log.csv")) if r["val_loss"]]
    if not rows:
        continue
    n += 1
    if abs(float(rows[0]["val_loss"]) - parent_end) > 5e-4:
        bad.append((run, rows[0]["val_loss"]))
print(f"[4] {n} upcycled runs start at the dense parent's val loss {parent_end:.4f}: {'all OK' if not bad else 'MISMATCH ' + str(bad)}")
assert not bad, "some upcycled run did not start at the parent's loss"
print("OK")
