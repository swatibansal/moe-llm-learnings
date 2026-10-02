"""
Training driver. Two phases, one script:

  # Phase 1 — train the dense ("linear") model from scratch
  python -m moe.train --run runs/dense --steps 1500

  # Phase 2a — control: keep training the dense model
  python -m moe.train --run runs/dense_cont --init runs/dense/ckpt.pkl --steps 1500

  # Phase 2b — upcycle the dense checkpoint into an MoE and keep training
  python -m moe.train --run runs/moe --init runs/dense/ckpt.pkl --upcycle --n-experts 4 --top-k 2 --steps 1500

Every run writes:  <run>/ckpt.pkl  (params+cfg+step)   <run>/log.csv   <run>/config.json
Resume: `--init <run>/ckpt.pkl` continues from the saved step counter (fresh optimizer state).
"""
from __future__ import annotations
import argparse, csv, json, math, os, pickle, time
import numpy as np
import jax, jax.numpy as jnp
from . import model as M
from .data import CharData

# ----------------------------------------------------------------------------
# hand-rolled AdamW (so nothing is hidden)
# ----------------------------------------------------------------------------

def adamw_init(p):
    z = jax.tree_util.tree_map(jnp.zeros_like, p)
    return {"m": z, "v": jax.tree_util.tree_map(jnp.zeros_like, p), "t": jnp.zeros((), jnp.int32)}


def adamw_update(p, g, st, lr, b1=0.9, b2=0.95, eps=1e-8, wd=0.1, clip=1.0):
    gnorm = jnp.sqrt(sum(jnp.sum(x * x) for x in jax.tree_util.tree_leaves(g)))
    g = jax.tree_util.tree_map(lambda x: x * jnp.minimum(1.0, clip / (gnorm + 1e-6)), g)
    t = st["t"] + 1
    m = jax.tree_util.tree_map(lambda m_, g_: b1 * m_ + (1 - b1) * g_, st["m"], g)
    v = jax.tree_util.tree_map(lambda v_, g_: b2 * v_ + (1 - b2) * g_ * g_, st["v"], g)
    bc1, bc2 = 1 - b1 ** t, 1 - b2 ** t

    def upd(p_, m_, v_):
        step = (m_ / bc1) / (jnp.sqrt(v_ / bc2) + eps)
        decay = wd if p_.ndim >= 2 else 0.0          # no decay on biases / LN gains
        return p_ - lr * (step + decay * p_)

    return jax.tree_util.tree_map(upd, p, m, v), {"m": m, "v": v, "t": t}, gnorm


def lr_at(step, total, base, warmup, final):
    if step < warmup:
        return base * (step + 1) / warmup
    frac = (step - warmup) / max(1, total - warmup)
    return final + 0.5 * (base - final) * (1 + math.cos(math.pi * frac))

# ----------------------------------------------------------------------------

def save_ckpt(path, params, cfg, step, opt=None, rng=None, phase_step=0):
    """Full training state, so a resumed run continues exactly (optimizer moments,
    batch RNG, LR-schedule position) — not just the weights."""
    d = {"params": jax.device_get(params), "cfg": cfg, "step": step, "phase_step": phase_step}
    if opt is not None:
        d["opt"] = jax.device_get(opt)
    if rng is not None:
        d["rng_state"] = rng.bit_generator.state
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(d, f)
    os.replace(tmp, path)


def load_ckpt(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    d["params"] = jax.tree_util.tree_map(jnp.asarray, d["params"])
    if "opt" in d:
        d["opt"] = jax.tree_util.tree_map(jnp.asarray, d["opt"])
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", default="data/input.txt")
    ap.add_argument("--init", default=None, help="checkpoint to start from (weights only; fresh optimizer)")
    ap.add_argument("--resume", action="store_true", help="continue <run>/ckpt.pkl with full state (opt, rng, schedule)")
    ap.add_argument("--max-seconds", type=float, default=0, help="stop (after saving) once this much wall time passed; 0 = no limit")
    ap.add_argument("--upcycle", action="store_true", help="convert --init dense ckpt into MoE")
    ap.add_argument("--n-experts", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--aux-coef", type=float, default=0.01)
    ap.add_argument("--router-std", type=float, default=1e-3,
                    help="router init std at --upcycle (0 = exact ties; only the aux loss can break symmetry)")
    ap.add_argument("--gate-grad", choices=["renorm", "stopgrad"], default="renorm",
                    help="how the top-k gate renormalisation is differentiated (see model.moe_ffn); "
                         "'stopgrad' lets the router learn with top-1. Stored in cfg at --upcycle time.")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-final", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-seed", type=int, default=1234, help="batch order; keep equal across compared runs")
    # architecture (only used when training from scratch)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--d-ff", type=int, default=512)
    ap.add_argument("--block", type=int, default=128)
    args = ap.parse_args()

    os.makedirs(args.run, exist_ok=True)
    key = jax.random.PRNGKey(args.seed)
    data = CharData(args.data, block=args.block)
    ckpt_path = os.path.join(args.run, "ckpt.pkl")
    done_path = os.path.join(args.run, "DONE")

    resumed = None
    if args.resume and os.path.exists(ckpt_path):
        resumed = load_ckpt(ckpt_path)
        params, cfg, step0 = resumed["params"], resumed["cfg"], resumed["step"]
        print(f"resuming {ckpt_path} @ step {step0} (phase step {resumed['phase_step']}/{args.steps})")
        if resumed["phase_step"] >= args.steps:
            print("phase already complete"); open(done_path, "w").write(str(step0)); return
    elif args.init:
        d = load_ckpt(args.init)
        params, cfg, step0 = d["params"], d["cfg"], d["step"]
        print(f"loaded {args.init} @ step {step0}  ({'MoE' if M.is_moe(params) else 'dense'})")
        if args.upcycle:
            assert not M.is_moe(params), "already an MoE"
            key, sub = jax.random.split(key)
            params = M.upcycle_dense_to_moe(params, args.n_experts, sub, router_std=args.router_std)
            cfg = {**cfg, "n_experts": args.n_experts, "top_k": args.top_k, "aux_coef": args.aux_coef,
                   "gate_grad": args.gate_grad, "router_std": args.router_std}
            print(f"upcycled dense MLPs -> MoE with E={args.n_experts}, top-k={args.top_k}, "
                  f"gate_grad={args.gate_grad}, router_std={args.router_std:g}")
    else:
        cfg = {"vocab": data.vocab, "block": args.block, "d_model": args.d_model, "n_head": args.n_head,
               "n_layer": args.n_layer, "d_ff": args.d_ff}
        key, sub = jax.random.split(key)
        params = M.init_dense(sub, cfg)
        step0 = 0
    if M.is_moe(params):
        cfg.setdefault("aux_coef", args.aux_coef)
    assert cfg["vocab"] == data.vocab, "checkpoint vocab != data vocab"

    n_total = M.count_params(params)
    n_active = M.count_active_params(params, cfg) if M.is_moe(params) else n_total
    print(f"params: total={n_total:,}  active/token={n_active:,}")
    json.dump({"args": vars(args), "cfg": cfg, "params_total": n_total, "params_active": n_active},
              open(os.path.join(args.run, "config.json"), "w"), indent=2)

    @jax.jit
    def train_step(params, opt, x, y, lr):
        (loss, (ce, aux, loads)), g = jax.value_and_grad(M.loss_fn, has_aux=True)(params, x, y, cfg)
        params, opt, gnorm = adamw_update(params, g, opt, lr)
        return params, opt, ce, aux, gnorm, loads

    @jax.jit
    def eval_step(params, x, y):
        _, (ce, aux, loads) = M.loss_fn(params, x, y, cfg)
        return ce, loads

    def evaluate(params):
        ces, loads = [], []
        for x, y in data.val_batches(args.batch, args.eval_batches):
            ce, ld = eval_step(params, x, y)
            ces.append(float(ce)); loads.append(ld)
        load = None
        if loads and loads[0]:
            load = np.mean(np.stack([np.stack([np.asarray(l) for l in ld]) for ld in loads]), 0)  # (L,E)
        return float(np.mean(ces)), load

    rng = np.random.default_rng(args.data_seed)
    if resumed:
        opt = resumed["opt"]
        rng.bit_generator.state = resumed["rng_state"]
        i0 = resumed["phase_step"]
        phase_start = step0 - i0                      # global step at which this phase began
        logf = open(os.path.join(args.run, "log.csv"), "a", newline="")
        log = csv.writer(logf)
    else:
        opt = adamw_init(params)
        i0, phase_start = 0, step0
        logf = open(os.path.join(args.run, "log.csv"), "w", newline="")
        log = csv.writer(logf)
        log.writerow(["step", "train_loss", "val_loss", "aux_loss", "lr", "grad_norm", "sec"])
        # loss *before* any update: for an upcycled run this must equal the dense parent's loss
        v0, load0 = evaluate(params)
        print(f"step {step0:5d}  val {v0:.4f}   (initial, no updates yet)")
        log.writerow([step0, "", f"{v0:.4f}", "", "", "", 0]); logf.flush()
        if load0 is not None:
            print("  expert load per layer (fraction of routing assignments):\n  " + "\n  ".join(
                " ".join(f"{v:.2f}" for v in row) for row in load0))
        save_ckpt(ckpt_path, params, cfg, step0, opt, rng, 0)

    t0 = time.time()
    run_ce = None
    finished = False
    for i in range(i0, args.steps):
        step = phase_start + i + 1
        lr = lr_at(i, args.steps, args.lr, args.warmup, args.lr_final)
        x, y = data.train_batch(rng, args.batch)
        params, opt, ce, aux, gnorm, loads = train_step(params, opt, x, y, lr)
        ce = float(ce)
        run_ce = ce if run_ce is None else 0.9 * run_ce + 0.1 * ce
        last = i == args.steps - 1
        out_of_time = args.max_seconds and (time.time() - t0) > args.max_seconds
        if step % args.eval_every == 0 or last or out_of_time:
            v, load = evaluate(params)
            dt = time.time() - t0
            extra = f"  aux {float(aux):.3f}" if M.is_moe(params) else ""
            print(f"step {step:5d}  train {run_ce:.4f}  val {v:.4f}{extra}  lr {lr:.2e}  |g| {float(gnorm):.2f}  {dt:6.0f}s", flush=True)
            log.writerow([step, f"{run_ce:.4f}", f"{v:.4f}", f"{float(aux):.4f}", f"{lr:.2e}", f"{float(gnorm):.3f}", f"{dt:.0f}"])
            logf.flush()
            if load is not None and (step % (args.eval_every * 5) == 0 or last):
                print("  expert load per layer:\n  " + "\n  ".join(" ".join(f"{v:.2f}" for v in row) for row in load))
            save_ckpt(ckpt_path, params, cfg, step, opt, rng, i + 1)
        if last:
            finished = True
        if out_of_time and not last:
            print(f"time budget hit at phase step {i + 1}/{args.steps}; state saved — rerun with --resume", flush=True)
            logf.close()
            return

    logf.close()
    open(done_path, "w").write(str(phase_start + args.steps))

    # a taste of what it learned
    key, sub = jax.random.split(key)
    ids = M.generate(params, cfg, sub, data.encode("ROMEO:\n"), 300)
    sample = data.decode(ids)
    open(os.path.join(args.run, "sample.txt"), "w").write(sample)
    print("\n--- sample ---\n" + sample + "\n--------------")


if __name__ == "__main__":
    main()
