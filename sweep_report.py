"""Tabulate the E/k sweep (run_sweep.sh) against the dense control. Usage: python sweep_report.py [runs_dir]

Rows: runs/dense_cont, runs/moe (E=4,k=2) and every runs/moe_e*_k*. All share the dense parent,
the phase-2 schedule and --data-seed 1234, so val loss at the common final step is comparable.
Also writes runs/sweep.png: val loss at the final step vs active params per token."""
import csv, glob, json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = sys.argv[1] if len(sys.argv) > 1 else "runs"


def load(run):
    p = os.path.join(root, run, "log.csv")
    if not os.path.exists(p):
        return None
    rows = [r for r in csv.DictReader(open(p)) if r["val_loss"]]
    cfg = json.load(open(os.path.join(root, run, "config.json")))
    c = cfg["cfg"]
    return {"run": run, "E": c.get("n_experts", 1), "k": c.get("top_k", 1), "gate": c.get("gate_grad", "renorm"),
            "aux_coef": c.get("aux_coef", 0.01), "router_std": c.get("router_std", 1e-3), "bias_gamma": c.get("bias_gamma", 0.0),
            "total": cfg["params_total"], "active": cfg["params_active"],
            "step0": float(rows[0]["val_loss"]), "step": int(rows[-1]["step"]), "val": float(rows[-1]["val_loss"]),
            "aux": float(rows[-1]["aux_loss"] or 0), "done": os.path.exists(os.path.join(root, run, "DONE"))}


names = ["dense_cont", "moe"] + sorted(os.path.basename(p) for p in glob.glob(os.path.join(root, "moe_e*_k*")))
rows = [r for r in (load(n) for n in names) if r]
ctrl = next((r for r in rows if r["run"] == "dense_cont"), None)

print("| run | E | k | gate | params total / active | val @ step | MoE − dense | aux | status |")
print("|---|---|---|---|---|---|---|---|---|")
for r in rows:
    gap = "" if ctrl is None or r is ctrl or r["step"] != ctrl["step"] else f"{r['val'] - ctrl['val']:+.4f}"
    E = "–" if r["run"] == "dense_cont" else r["E"]
    k = "–" if r["run"] == "dense_cont" else r["k"]
    gate = "–" if r["run"] == "dense_cont" else r["gate"]
    aux = "" if r["run"] == "dense_cont" else f"{r['aux']:.3f}"
    print(f"| `{r['run']}` | {E} | {k} | {gate} | {r['total']/1e6:.2f}M / {r['active']/1e6:.2f}M | "
          f"{r['val']:.4f} @ {r['step']} | {gap} | {aux} | {'DONE' if r['done'] else 'running'} |")
bad = [r["run"] for r in rows if abs(r["step0"] - 1.6210) > 1e-3]
print("\nstep-0 val == dense parent (1.6210) for all arms:", "yes" if not bad else f"NO: {bad}")

fin = [r for r in rows if r["done"]]
if fin:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in fin:
        sg = r["gate"] == "stopgrad"
        lab = "dense (control)" if r["run"] == "dense_cont" else f"E={r['E']}, top-{r['k']}" + (" (stopgrad)" if sg else "")
        if r["run"] != "dense_cont":  # router-ablation arms: mark the non-default knob
            if r["aux_coef"] != 0.01: lab += f" aux={r['aux_coef']:g}"
            if r["router_std"] != 1e-3: lab += f" std={r['router_std']:g}"
            if r["bias_gamma"]: lab += f" bias={r['bias_gamma']:g}"
        ax.scatter(r["active"] / 1e6, r["val"], s=60, marker="^" if sg else "o",
                   color="tab:blue" if r["run"] == "dense_cont" else ("tab:green" if sg else "tab:red"))
        ax.annotate(lab, (r["active"] / 1e6, r["val"]), textcoords="offset points", xytext=(6, 4), fontsize=8)
    if ctrl and ctrl["done"]:
        ax.axhline(ctrl["val"], color="tab:blue", ls=":", lw=1)
    ax.set_xlabel("active params per token (M)"); ax.set_ylabel(f"val loss @ step {fin[0]['step']} (nats/char)")
    ax.set_title("Upcycled MoE sweep: same dense parent, same batches, 1500 steps"); ax.grid(alpha=.3)
    out = os.path.join(root, "sweep.png")
    plt.tight_layout(); plt.savefig(out, dpi=130); print("wrote", out)
