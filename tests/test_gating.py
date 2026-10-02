"""Gating tests for moe_ffn. Run:  python tests/test_gating.py   (plain asserts, no pytest needed)

Background (sweep finding 2026-09-30): with top-k gates renormalised as topv / topv.sum(), the k=1 gate is
identically 1, so d(gate)/d(router) == 0 and the router gets no LM-loss gradient. cfg["gate_grad"] == "stopgrad"
divides by stop_gradient(sum) instead: same forward values (so upcycling stays exactly function-preserving),
but the selected probability now carries gradient to the router for every k.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import jax, jax.numpy as jnp
import numpy as np
from moe import model as M

CFG = {"vocab": 11, "block": 8, "d_model": 16, "n_head": 2, "n_layer": 2, "d_ff": 32}
key = jax.random.PRNGKey(0)
dense = M.init_dense(key, CFG)
# make the dense model non-trivial so gradients are not degenerate
dense = jax.tree_util.tree_map(lambda a: a * 5.0 if a.ndim >= 2 else a, dense)
idx = jax.random.randint(jax.random.PRNGKey(1), (3, 8), 0, CFG["vocab"])
tgt = jnp.roll(idx, -1, axis=1)
dense_logits = M.forward(dense, idx, CFG)[0]


def moe_cfg(E, k, gate_grad=None):
    c = {**CFG, "n_experts": E, "top_k": k, "aux_coef": 0.0}
    if gate_grad is not None:
        c["gate_grad"] = gate_grad
    return c


def router_grad_norm(params, cfg):
    g = jax.grad(lambda p: M.loss_fn(p, idx, tgt, cfg)[0])(params)
    return float(sum(jnp.sum(b["moe"]["router"] ** 2) for b in g["blocks"]) ** 0.5)


def test_upcycle_is_function_preserving_for_both_gate_modes():
    for E, k in [(2, 1), (4, 1), (4, 2), (8, 2)]:
        moe = M.upcycle_dense_to_moe(dense, E, jax.random.PRNGKey(2))
        for mode in ("renorm", "stopgrad"):
            out = M.forward(moe, idx, moe_cfg(E, k, mode))[0]
            err = float(jnp.max(jnp.abs(out - dense_logits)))
            assert err < 1e-4, f"E={E} k={k} {mode}: max |dense - moe| = {err}"


def test_stopgrad_forward_matches_renorm_forward_after_training_signal():
    # values must be identical for any params, not just at the symmetric upcycle point
    moe = M.upcycle_dense_to_moe(dense, 4, jax.random.PRNGKey(2), router_std=0.5)
    moe["blocks"][0]["moe"]["w1"] = moe["blocks"][0]["moe"]["w1"] * jnp.arange(1, 5, dtype=jnp.float32)[:, None, None]
    a = M.forward(moe, idx, moe_cfg(4, 2, "renorm"))[0]
    b = M.forward(moe, idx, moe_cfg(4, 2, "stopgrad"))[0]
    assert float(jnp.max(jnp.abs(a - b))) < 1e-5


def test_top1_renorm_router_gets_no_lm_gradient():
    moe = M.upcycle_dense_to_moe(dense, 4, jax.random.PRNGKey(2))
    assert router_grad_norm(moe, moe_cfg(4, 1, "renorm")) < 1e-6  # measured 7e-9: float noise, vs ~1e-2 for top-2


def test_top1_stopgrad_router_gets_lm_gradient():
    moe = M.upcycle_dense_to_moe(dense, 4, jax.random.PRNGKey(2))
    # at the exact upcycle point all experts agree so the gradient is ~0 by symmetry; perturb one expert
    moe["blocks"][0]["moe"]["w2"] = moe["blocks"][0]["moe"]["w2"].at[0].multiply(1.5)
    assert router_grad_norm(moe, moe_cfg(4, 1, "stopgrad")) > 1e-4


def test_missing_gate_grad_key_defaults_to_renorm():
    moe = M.upcycle_dense_to_moe(dense, 4, jax.random.PRNGKey(2))
    assert router_grad_norm(moe, moe_cfg(4, 1)) < 1e-6  # measured 7e-9: float noise, vs ~1e-2 for top-2


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t(); print("PASS", t.__name__)
        except Exception as e:  # noqa
            failed += 1; print("FAIL", t.__name__, "->", type(e).__name__, e)
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
