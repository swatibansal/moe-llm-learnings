"""Plot val loss for dense -> {dense_cont, moe}. Usage: python plot.py [runs_dir]
Left: whole run. Right: zoom on phase 2 (after the upcycle) where the comparison lives."""
import csv, json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = sys.argv[1] if len(sys.argv) > 1 else "runs"


def read(run):
    p = os.path.join(root, run, "log.csv")
    if not os.path.exists(p):
        return None
    rows = [r for r in csv.DictReader(open(p)) if r["val_loss"]]
    return [int(r["step"]) for r in rows], [float(r["val_loss"]) for r in rows]


# run -> (label, colour, linestyle). Runs whose log.csv is missing are skipped, so the
# *_long arms (run_long.sh, 5000-step phase 2) only appear once they exist.
styles = {"dense": ("Phase 1: dense", "k", "-"),
          "dense_cont": ("Phase 2a: dense, continued (control), 1500 steps", "tab:blue", "-"),
          "moe": ("Phase 2b: upcycled MoE, continued, 1500 steps", "tab:red", "-"),
          "dense_cont_long": ("Phase 2a-long: dense control, 5000 steps", "tab:blue", "--"),
          "moe_long": ("Phase 2b-long: upcycled MoE, 5000 steps", "tab:red", "--")}
curves, summary = {}, {}
for run, (label, c, ls) in styles.items():
    d = read(run)
    if not d:
        continue
    s, v = d
    cfg = json.load(open(os.path.join(root, run, "config.json")))
    label += f"  [{cfg['params_total']/1e6:.2f}M total, {cfg['params_active']/1e6:.2f}M active/token]"
    curves[run] = (s, v, label, c, ls)
    summary[run] = {"start": (s[0], v[0]), "end": (s[-1], v[-1]), "min": min(v)}

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.1, 1]})
for run, (s, v, label, c, ls) in curves.items():
    ax.plot(s, v, color=c, lw=2, ls=ls, label=label)
    if run != "dense":
        ax2.plot(s, v, color=c, lw=2, ls=ls, marker="o", ms=3, label=label)
if "dense" in summary:
    x0 = summary["dense"]["end"][0]
    for a in (ax, ax2):
        a.axvline(x0, color="gray", ls=":", lw=1)
    ax.text(x0, ax.get_ylim()[1], " upcycle: dense MLP -> E identical experts + router", va="top", color="gray", fontsize=8)
ax.set_xlabel("training step"); ax.set_ylabel("validation loss (nats/char)")
ax.set_title("Full run"); ax.grid(alpha=.3); ax.legend(fontsize=7, loc="upper right")
ax2.set_xlabel("training step"); ax2.set_title("Phase 2 zoom: same start weights, same batches")
ax2.grid(alpha=.3); ax2.legend(fontsize=7)
fig.suptitle("Dense -> Sparse-upcycled MoE on Shakespeare (char-level GPT, pure JAX)")
out = os.path.join(root, "loss_curves.png")
plt.tight_layout(); plt.savefig(out, dpi=130)
print("wrote", out)
for k, v in summary.items():
    print(f"{k:15s} start {v['start'][1]:.4f} @ {v['start'][0]:5d}  ->  end {v['end'][1]:.4f} @ {v['end'][0]:5d}   (min {v['min']:.4f})")
for ctrl, moe in (("dense_cont", "moe"), ("dense_cont_long", "moe_long")):
    if ctrl in summary and moe in summary and summary[ctrl]["end"][0] == summary[moe]["end"][0]:
        gap = summary[moe]["end"][1] - summary[ctrl]["end"][1]
        print(f"{moe} - {ctrl} at step {summary[moe]['end'][0]}: {gap:+.4f} nats/char")

# ---- headline figure: dense control vs E=4 default vs E=8 recommended (ablation #3b), phase 2 only ----
headline = {"dense_cont": ("dense, continued (control)", "tab:blue"),
            "moe": ("upcycled MoE E=4 top-2, defaults (aux 0.01, router_std 1e-3)", "tab:red"),
            "moe_e8_k2_aux0p1_std0p01": ("upcycled MoE E=8 top-2, recommended (aux 0.1, router_std 1e-2)", "tab:green")}
hc = {}
for run, (label, c) in headline.items():
    d = read(run)
    if d and os.path.exists(os.path.join(root, run, "config.json")):
        cfg = json.load(open(os.path.join(root, run, "config.json")))
        hc[run] = (*d, f"{label}  [{cfg['params_total']/1e6:.2f}M total, {cfg['params_active']/1e6:.2f}M active/token]", c)
if len(hc) == 3:
    fig, ax = plt.subplots(figsize=(8, 5))
    for run, (s, v, label, c) in hc.items():
        ax.plot(s, v, color=c, lw=2, marker="o", ms=3, label=label)
    for run, (s, v, label, c) in hc.items():
        ax.annotate(f"{v[-1]:.4f}", (s[-1], v[-1]), textcoords="offset points", xytext=(6, 0), color=c, fontsize=9, va="center")
    ax.set_xlabel("training step"); ax.set_ylabel("validation loss (nats/char)")
    ax.set_title("Same dense parent @2000, same batches: dense vs upcycled MoE (1500 more steps)")
    ax.grid(alpha=.3); ax.legend(fontsize=8)
    out2 = os.path.join(root, "headline.png")
    plt.tight_layout(); plt.savefig(out2, dpi=130); print("wrote", out2)
