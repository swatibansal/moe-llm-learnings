"""Router diagnostics for trained MoE checkpoints. Usage: python router_analysis.py runs/moe_e8_k2 [runs/... ...]

For each run: evaluates the model on fixed validation windows and reports, per MoE layer,
  * load        fraction of routing assignments per expert (what train.py prints every 500 steps)
  * max load    largest share taken by one expert (1/E = perfectly balanced; -> 1 = collapse)
  * dead        experts receiving < 1% of assignments
  * H(P)/ln E   normalised entropy of the mean router distribution (1 = uniform)
  * router RMS  weight scale of the router matrix (init std is in cfg["router_std"])
and the val loss over the same windows, so arms can be compared on identical data.
Prints a markdown summary table at the end (one row per run).
"""
import json, os, pickle, sys
import numpy as np
import jax, jax.numpy as jnp
from moe import model as M
from moe.data import CharData

N_BATCHES = int(os.environ.get("N_BATCHES", 20))   # same default as train.py --eval-batches
BATCH = int(os.environ.get("BATCH", 32))


def load_run(run):
    ck = pickle.load(open(os.path.join(run, "ckpt.pkl"), "rb"))
    cfg = json.load(open(os.path.join(run, "config.json")))
    return ck["params"], ck["cfg"], cfg


def route_stats(params, cfg, data):
    """Per-layer expert loads and mean router probs over N_BATCHES fixed val windows."""
    E, k = cfg["n_experts"], cfg["top_k"]
    fwd = jax.jit(lambda p, x, y: M.loss_fn(p, x, y, cfg))
    # re-derive router probabilities layer by layer (forward() only returns loads)
    def probs_by_layer(p, idx):
        B, T = idx.shape
        x = p["tok_emb"][idx] + p["pos_emb"][:T][None]
        out = []
        for blk in p["blocks"]:
            x = x + M.attention(blk["attn"], M.layer_norm(x, **blk["ln1"]), cfg["n_head"])
            h = M.layer_norm(x, **blk["ln2"])
            pr = jax.nn.softmax(h @ blk["moe"]["router"], axis=-1)
            out.append(pr.reshape(-1, E))
            y, _, _ = M.moe_ffn(blk["moe"], h, k, cfg.get("gate_grad", "renorm"))
            x = x + y
        return out
    pbl = jax.jit(probs_by_layer)
    L = len(params["blocks"])
    loads = np.zeros((L, E)); P = np.zeros((L, E)); losses = []
    for x, y in data.val_batches(BATCH, N_BATCHES):
        loss, (ce, aux, f) = fwd(params, x, y)
        losses.append(float(ce))
        loads += np.stack([np.asarray(fi) for fi in f])
        P += np.stack([np.asarray(pr).mean(0) for pr in pbl(params, x)])
    return loads / N_BATCHES, P / N_BATCHES, float(np.mean(losses))


def main(runs):
    rows = []
    data = None
    for run in runs:
        params, cfg, meta = load_run(run)
        if not M.is_moe(params):
            print(f"{run}: dense, skipped"); continue
        if data is None:
            data = CharData(meta["args"]["data"], block=cfg["block"])
        loads, P, val = route_stats(params, cfg, data)
        E = cfg["n_experts"]
        H = -(P * np.log(P + 1e-12)).sum(-1) / np.log(E)
        rms = [float(jnp.sqrt((b["moe"]["router"] ** 2).mean())) for b in params["blocks"]]
        print(f"\n== {run}   E={E} top-{cfg['top_k']} aux_coef={cfg.get('aux_coef')} router_std={cfg.get('router_std', '1e-3 (default)')} "
              f"gate={cfg.get('gate_grad', 'renorm')}   val={val:.4f} (over {N_BATCHES} fixed val batches)")
        print("  layer  max-load  dead  H(P)/lnE  routerRMS   loads")
        for l in range(len(loads)):
            dead = int((loads[l] < 0.01).sum())
            print(f"  {l:5d}  {loads[l].max():8.2f}  {dead:4d}  {H[l]:8.2f}  {rms[l]:9.2e}   " + " ".join(f"{v:.2f}" for v in loads[l]))
        rows.append((os.path.basename(run), cfg.get("aux_coef"), cfg.get("router_std", 1e-3), val,
                     loads.max(1).mean(), loads.max(), int((loads < 0.01).sum()), H.mean()))
    if rows:
        print("\n| run | aux_coef | router_std | val (fixed batches) | mean max-load | worst max-load | dead experts (all layers) | mean H(P)/lnE |")
        print("|---|---|---|---|---|---|---|---|")
        for r in rows:
            print(f"| `{r[0]}` | {r[1]} | {r[2]:g} | {r[3]:.4f} | {r[4]:.2f} | {r[5]:.2f} | {r[6]} | {r[7]:.2f} |")


if __name__ == "__main__":
    main(sys.argv[1:] or ["runs/moe"])
