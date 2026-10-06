"""Bounded-score variant of the selection bias. Run: python tests/test_bias_bounded.py

Finding (run moe_e8_k2_bias0p001): adding the bias to softmax LOGITS fails once the router grows confident —
logit gaps become larger than any bias the ±gamma walk can reach, so overloaded experts stay overloaded.
DeepSeek-V3 adds the bias to BOUNDED affinities (sigmoid in [0,1]), so a bias of order 1 can always flip a choice.
cfg["bias_on"] == "probs" adds the bias to the softmax probabilities (bounded in [0,1]) for selection;
"logits" (default) is the original behaviour. Gate weights are unchanged in both modes.
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
E, K = 4, 2
dense_logits = M.forward(dense, idx, CFG)[0]


def cfg(bias_on=None):
    c = {**CFG, "n_experts": E, "top_k": K, "aux_coef": 0.0}
    if bias_on is not None:
        c["bias_on"] = bias_on
    return c


def confident_router_moe():
    """Router scaled so logit gaps are >> 2 (a very confident, collapsed router)."""
    moe = M.upcycle_dense_to_moe(dense, E, jax.random.PRNGKey(2), router_std=1e-3)
    for blk in moe["blocks"]:
        blk["moe"]["router"] = blk["moe"]["router"] * 5000.0
    return moe


def test_bias_of_two_on_probs_always_forces_selection_but_on_logits_does_not():
    moe = confident_router_moe()
    _, _, loads0 = M.forward(moe, idx, cfg())
    starved = int(jnp.argmin(loads0[0]))                      # an expert the confident router ignores in layer 0
    for blk in moe["blocks"]:
        blk["moe"]["sel_bias"] = jnp.zeros((E,)).at[starved].set(2.0)
    _, _, l_logits = M.forward(moe, idx, cfg("logits"))
    _, _, l_probs = M.forward(moe, idx, cfg("probs"))
    assert float(l_logits[0][starved]) < 0.5, "a bias of 2 should NOT overcome huge logit gaps"
    assert abs(float(l_probs[0][starved]) - 0.5) < 1e-6, "on probs (<=1) a bias of 2 must select it for every token (k=2 -> load 0.5)"


def test_probs_mode_is_function_preserving_with_arbitrary_bias():
    moe = M.upcycle_dense_to_moe(dense, E, jax.random.PRNGKey(2))
    for blk in moe["blocks"]:
        blk["moe"]["sel_bias"] = jnp.array([0.9, -0.4, 0.2, -0.7])
    out = M.forward(moe, idx, cfg("probs"))[0]
    assert float(jnp.max(jnp.abs(out - dense_logits))) < 1e-4


def test_missing_bias_on_key_means_logits():
    moe = confident_router_moe()
    for blk in moe["blocks"]:
        blk["moe"]["sel_bias"] = jnp.array([2.0, 0.0, 0.0, 0.0])
    a = M.forward(moe, idx, cfg())[2]
    b = M.forward(moe, idx, cfg("logits"))[2]
    assert all(np.allclose(np.asarray(x), np.asarray(y)) for x, y in zip(a, b))


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t(); print("PASS", t.__name__)
        except Exception as e:  # noqa
            failed += 1; print("FAIL", t.__name__, "->", type(e).__name__, str(e)[:140])
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
