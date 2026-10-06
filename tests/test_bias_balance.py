"""Auxiliary-loss-free load balancing (DeepSeek-V3 style) tests. Run: python tests/test_bias_balance.py

Each MoE block carries a per-expert selection bias `sel_bias` (E,). It is added to the router logits ONLY for
choosing the top-k experts; gate weights still come from the unbiased softmax, renormalised over the selected set.
After each training step the bias is nudged by a fixed step gamma: down for experts above the mean load, up for
those below (sign rule). It receives no gradient and is not counted as a model parameter.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import jax, jax.numpy as jnp
import numpy as np
from moe import model as M

CFG = {"vocab": 11, "block": 8, "d_model": 16, "n_head": 2, "n_layer": 2, "d_ff": 32}
dense = M.init_dense(jax.random.PRNGKey(0), CFG)
dense = jax.tree_util.tree_map(lambda a: a * 5.0 if a.ndim >= 2 else a, dense)
idx = jax.random.randint(jax.random.PRNGKey(1), (3, 8), 0, CFG["vocab"])
tgt = jnp.roll(idx, -1, axis=1)
E, K = 4, 2
CFG_MOE = {**CFG, "n_experts": E, "top_k": K, "aux_coef": 0.0}
dense_logits = M.forward(dense, idx, CFG)[0]


def upcycled(router_std=1e-3):
    return M.upcycle_dense_to_moe(dense, E, jax.random.PRNGKey(2), router_std=router_std)


def test_upcycle_adds_zero_selection_bias_per_layer():
    moe = upcycled()
    for blk in moe["blocks"]:
        b = blk["moe"]["sel_bias"]
        assert b.shape == (E,) and float(jnp.abs(b).max()) == 0.0


def test_any_bias_is_function_preserving_while_experts_are_identical():
    moe = upcycled()
    for blk in moe["blocks"]:
        blk["moe"]["sel_bias"] = jnp.array([3.0, -2.0, 0.5, -7.0])
    out = M.forward(moe, idx, CFG_MOE)[0]
    assert float(jnp.max(jnp.abs(out - dense_logits))) < 1e-4


def test_large_negative_bias_removes_an_expert_from_selection():
    moe = upcycled(router_std=0.1)
    for blk in moe["blocks"]:
        blk["moe"]["sel_bias"] = jnp.array([-100.0, 0.0, 0.0, 0.0])
    _, _, loads = M.forward(moe, idx, CFG_MOE)
    for f in loads:
        assert float(f[0]) == 0.0 and abs(float(f.sum()) - 1.0) < 1e-5


def test_bias_update_moves_overloaded_down_and_underloaded_up_by_gamma():
    moe = upcycled()
    loads = [jnp.array([0.5, 0.25, 0.25, 0.0]) for _ in moe["blocks"]]   # mean is 0.25
    new = M.update_sel_bias(moe, loads, gamma=0.01)
    for blk in new["blocks"]:
        b = np.asarray(blk["moe"]["sel_bias"])
        assert np.allclose(b, [-0.01, 0.0, 0.0, +0.01]), b
    # untouched when gamma == 0
    same = M.update_sel_bias(moe, loads, gamma=0.0)
    assert all(float(jnp.abs(b["moe"]["sel_bias"]).max()) == 0.0 for b in same["blocks"])


def test_selection_bias_is_not_counted_as_a_parameter():
    moe = upcycled()
    n_with = M.count_params(moe)
    stripped = jax.tree_util.tree_map(lambda a: a, moe)
    for blk in stripped["blocks"]:
        blk["moe"] = {k: v for k, v in blk["moe"].items() if k != "sel_bias"}
    assert n_with == M.count_params(stripped)
    assert M.count_active_params(moe, CFG_MOE) == M.count_active_params(stripped, CFG_MOE)


def test_selection_bias_receives_no_gradient():
    moe = upcycled(router_std=0.1)
    moe["blocks"][0]["moe"]["w2"] = moe["blocks"][0]["moe"]["w2"].at[0].multiply(1.5)
    g = jax.grad(lambda p: M.loss_fn(p, idx, tgt, CFG_MOE)[0])(moe)
    assert all(float(jnp.abs(b["moe"]["sel_bias"]).max()) == 0.0 for b in g["blocks"])


def test_old_checkpoints_without_bias_still_run():
    moe = upcycled()
    for blk in moe["blocks"]:
        del blk["moe"]["sel_bias"]
    out = M.forward(moe, idx, CFG_MOE)[0]
    assert float(jnp.max(jnp.abs(out - dense_logits))) < 1e-4


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t(); print("PASS", t.__name__)
        except Exception as e:  # noqa
            failed += 1; print("FAIL", t.__name__, "->", type(e).__name__, str(e)[:120])
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
