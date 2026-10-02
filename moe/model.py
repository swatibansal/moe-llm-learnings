"""
Tiny char-level GPT in pure JAX, with two kinds of feed-forward block:

  * dense MLP  ("Linear" model):  y = W2 · gelu(W1 · x)
  * MoE        (sparse upcycled): y = Σ_{e ∈ top-k} g_e(x) · MLP_e(x)

and a function `upcycle_dense_to_moe` that converts a trained dense checkpoint
into an MoE checkpoint *without changing its function* (Sparse Upcycling,
Komatsuzaki et al. 2022): every expert starts as a copy of the trained dense
MLP, and the router starts near-uniform, so the MoE's loss at step 0 equals the
dense model's loss. Training then breaks the symmetry and experts specialise.

Params are plain nested dicts/lists of jnp arrays (pytrees). No Flax, no optax —
everything you need to read is in this file and train.py.
"""
from __future__ import annotations
import math
import jax
import jax.numpy as jnp
import numpy as np

# ----------------------------------------------------------------------------
# init
# ----------------------------------------------------------------------------

def _normal(key, shape, std=0.02):
    return std * jax.random.normal(key, shape, dtype=jnp.float32)


def init_dense(key, cfg: dict) -> dict:
    """cfg keys: vocab, block, d_model, n_head, n_layer, d_ff"""
    V, T, D, L, H = cfg["vocab"], cfg["block"], cfg["d_model"], cfg["n_layer"], cfg["d_ff"]
    keys = jax.random.split(key, 3 + 4 * L)
    p = {
        "tok_emb": _normal(keys[0], (V, D)),
        "pos_emb": _normal(keys[1], (T, D)),
        "blocks": [],
        "ln_f": {"g": jnp.ones((D,)), "b": jnp.zeros((D,))},
        "head": _normal(keys[2], (D, V)),
    }
    # scale residual-branch output projections by 1/sqrt(2L) (GPT-2 trick)
    res_std = 0.02 / math.sqrt(2 * L)
    for i in range(L):
        k = keys[3 + 4 * i: 3 + 4 * i + 4]
        p["blocks"].append({
            "ln1": {"g": jnp.ones((D,)), "b": jnp.zeros((D,))},
            "attn": {"wqkv": _normal(k[0], (D, 3 * D)), "wo": _normal(k[1], (D, D), res_std)},
            "ln2": {"g": jnp.ones((D,)), "b": jnp.zeros((D,))},
            "mlp": {"w1": _normal(k[2], (D, H)), "b1": jnp.zeros((H,)),
                    "w2": _normal(k[3], (H, D), res_std), "b2": jnp.zeros((D,))},
        })
    return p


def upcycle_dense_to_moe(dense_params: dict, n_experts: int, key, router_std: float = 1e-3) -> dict:
    """Sparse upcycling: dense MLP  ->  n_experts identical copies + a router.

    With top-k gates renormalised to sum to 1 and identical experts, the MoE
    output is *exactly* the dense output regardless of which experts are picked,
    so the converted model's loss == the dense model's loss. router_std is tiny
    but non-zero so ties are broken and experts can start to diverge.
    """
    p = jax.tree_util.tree_map(lambda a: a, dense_params)  # shallow copy of the tree
    new_blocks = []
    for i, blk in enumerate(p["blocks"]):
        blk = dict(blk)
        mlp = blk.pop("mlp")
        D = mlp["w1"].shape[0]
        rk = jax.random.fold_in(key, i)
        blk["moe"] = {
            "router": _normal(rk, (D, n_experts), router_std),
            "w1": jnp.stack([mlp["w1"]] * n_experts),  # (E, D, H)
            "b1": jnp.stack([mlp["b1"]] * n_experts),  # (E, H)
            "w2": jnp.stack([mlp["w2"]] * n_experts),  # (E, H, D)
            "b2": jnp.stack([mlp["b2"]] * n_experts),  # (E, D)
        }
        new_blocks.append(blk)
    p["blocks"] = new_blocks
    return p


def count_params(p) -> int:
    return int(sum(a.size for a in jax.tree_util.tree_leaves(p)))


def count_active_params(p, cfg) -> int:
    """Params touched per token: MoE experts count only k of E."""
    total = 0
    for a_path, a in jax.tree_util.tree_leaves_with_path(p):
        path = "/".join(str(k) for k in a_path)
        if "moe" in path and "router" not in path:
            total += a.size * cfg["top_k"] // cfg["n_experts"]
        else:
            total += a.size
    return int(total)

# ----------------------------------------------------------------------------
# forward
# ----------------------------------------------------------------------------

def layer_norm(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / jnp.sqrt(var + eps) * g + b


def attention(p, x, n_head):
    B, T, D = x.shape
    hd = D // n_head
    qkv = x @ p["wqkv"]                                        # (B,T,3D)
    q, k, v = jnp.split(qkv, 3, axis=-1)
    q = q.reshape(B, T, n_head, hd).transpose(0, 2, 1, 3)      # (B,h,T,hd)
    k = k.reshape(B, T, n_head, hd).transpose(0, 2, 1, 3)
    v = v.reshape(B, T, n_head, hd).transpose(0, 2, 1, 3)
    att = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(hd)         # (B,h,T,T)
    mask = jnp.tril(jnp.ones((T, T), dtype=bool))
    att = jnp.where(mask, att, -1e9)
    att = jax.nn.softmax(att, axis=-1)
    y = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, D)
    return y @ p["wo"]


def dense_mlp(p, x):
    return jax.nn.gelu(x @ p["w1"] + p["b1"]) @ p["w2"] + p["b2"]


def moe_ffn(p, x, top_k: int, gate_grad: str = "renorm"):
    """Top-k routed mixture of experts.

    Educational 'dense-masked' implementation: every expert is evaluated on
    every token and non-selected experts are multiplied by a zero gate. This is
    mathematically identical to real sparse dispatch (gather tokens -> expert
    -> scatter back) but costs E× the FLOPs of one expert. Fine for a ~1M-param
    toy; a production MoE would dispatch. Returns (y, aux_loss, expert_load).

    gate_grad controls how the renormalised gate topv / sum(topv) is differentiated:
      "renorm"   — plain autodiff. With k=1 the gate is identically 1, so the router
                   gets NO gradient from the LM loss (only from the aux loss). Original behaviour.
      "stopgrad" — divide by stop_gradient(sum): identical forward values (upcycling stays exactly
                   function-preserving) but gradient flows through the selected prob for every k.
    """
    B, T, D = x.shape
    E = p["router"].shape[1]
    logits = x @ p["router"]                                   # (B,T,E)
    probs = jax.nn.softmax(logits, axis=-1)
    topv, topi = jax.lax.top_k(probs, top_k)                   # (B,T,k)
    denom = topv.sum(-1, keepdims=True)
    if gate_grad == "stopgrad":
        denom = jax.lax.stop_gradient(denom)
    elif gate_grad != "renorm":
        raise ValueError(f"gate_grad must be 'renorm' or 'stopgrad', got {gate_grad!r}")
    gates_k = topv / denom                                     # renormalise selected gates to sum to 1
    onehot = jax.nn.one_hot(topi, E)                           # (B,T,k,E)
    gates = (onehot * gates_k[..., None]).sum(-2)              # (B,T,E) sparse: zeros for unselected

    h = jax.nn.gelu(jnp.einsum("btd,edh->bteh", x, p["w1"]) + p["b1"])   # (B,T,E,H)
    y_e = jnp.einsum("bteh,ehd->bted", h, p["w2"]) + p["b2"]             # (B,T,E,D)
    y = jnp.einsum("bte,bted->btd", gates, y_e)

    # Switch-Transformer load-balancing loss: E * Σ_e f_e · P_e
    # f_e = fraction of routing *assignments* that went to e (sums to 1 over e),
    # P_e = mean router prob of e. Minimum (=1.0) when both are uniform.
    mask = onehot.sum(-2)                                      # (B,T,E) 0/1
    f = mask.mean((0, 1)) / top_k                              # (E,)
    P = probs.mean((0, 1))                                     # (E,)
    aux = E * jnp.sum(f * P)
    return y, aux, f


def forward(p, idx, cfg):
    """idx: (B,T) int32 -> logits (B,T,V), aux_loss (scalar), loads list[(E,)]"""
    B, T = idx.shape
    x = p["tok_emb"][idx] + p["pos_emb"][:T][None]
    aux_total = 0.0
    loads = []
    for blk in p["blocks"]:
        x = x + attention(blk["attn"], layer_norm(x, **blk["ln1"]), cfg["n_head"])
        h = layer_norm(x, **blk["ln2"])
        if "moe" in blk:
            y, aux, f = moe_ffn(blk["moe"], h, cfg["top_k"], cfg.get("gate_grad", "renorm"))
            aux_total = aux_total + aux
            loads.append(f)
        else:
            y = dense_mlp(blk["mlp"], h)
        x = x + y
    x = layer_norm(x, **p["ln_f"])
    if loads:
        aux_total = aux_total / len(loads)                     # mean over MoE layers -> min 1.0
    return x @ p["head"], aux_total, loads


def loss_fn(p, idx, targets, cfg):
    logits, aux, loads = forward(p, idx, cfg)
    logp = jax.nn.log_softmax(logits, axis=-1)
    ce = -jnp.take_along_axis(logp, targets[..., None], axis=-1).mean()
    total = ce + cfg.get("aux_coef", 0.0) * aux
    return total, (ce, aux, loads)


def is_moe(p) -> bool:
    return "moe" in p["blocks"][0]

# ----------------------------------------------------------------------------
# sampling (for eyeballing what the model learned)
# ----------------------------------------------------------------------------

def generate(p, cfg, key, prompt_ids, n_new: int, temperature=0.8):
    """Left-pads the context to a fixed `block` length so the jitted forward compiles once."""
    T = cfg["block"]
    fwd = jax.jit(lambda p_, ctx: forward(p_, ctx, cfg)[0])
    ids = list(prompt_ids)
    for _ in range(n_new):
        ctx = ids[-T:]
        n = len(ctx)
        padded = jnp.asarray([0] * (T - n) + ctx, dtype=jnp.int32)[None]
        logits = fwd(p, padded)                                # causal mask => padding only affects earlier positions
        key, sub = jax.random.split(key)
        nxt = jax.random.categorical(sub, logits[0, -1] / temperature)
        ids.append(int(nxt))
    return ids
