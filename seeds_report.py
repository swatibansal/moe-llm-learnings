"""Error bars for the headline claim. Usage: python seeds_report.py

Paired seeds (run_seeds.sh): within a pair both arms share --data-seed (identical batches), so the per-pair
gap (MoE − dense) is the quantity with the least noise. Reports per-pair end val loss and gap, plus mean ± std
(sample std, n−1) across pairs. Pair 0 is the original data-seed-1234 pair.
"""
import csv, os
import numpy as np

PAIRS = [("dense_cont", "moe_e8_k2_aux0p1_std0p01", "1234 (orig)"),
         ("dense_cont_s1", "moe_e8_rec_s1", "1"),
         ("dense_cont_s2", "moe_e8_rec_s2", "2")]


def end_val(run):
    p = f"runs/{run}/log.csv"
    if not os.path.exists(p) or not os.path.exists(f"runs/{run}/DONE"):
        return None
    rows = [r for r in csv.DictReader(open(p)) if r["val_loss"]]
    return float(rows[-1]["val_loss"]), int(rows[-1]["step"])


d_vals, m_vals, gaps = [], [], []
print("| data-seed | dense control @step | MoE E=8 recommended @step | MoE − dense |")
print("|---|---|---|---|")
for d, m, s in PAIRS:
    dv, mv = end_val(d), end_val(m)
    if dv is None or mv is None:
        print(f"| {s} | {'—' if dv is None else f'{dv[0]:.4f} @{dv[1]}'} | {'—' if mv is None else f'{mv[0]:.4f} @{mv[1]}'} | (incomplete) |")
        continue
    assert dv[1] == mv[1], f"pair {s}: different end steps"
    d_vals.append(dv[0]); m_vals.append(mv[0]); gaps.append(mv[0] - dv[0])
    print(f"| {s} | {dv[0]:.4f} @{dv[1]} | {mv[0]:.4f} @{mv[1]} | {gaps[-1]:+.4f} |")

if len(gaps) >= 2:
    sd = lambda a: np.std(a, ddof=1)
    print(f"\nn = {len(gaps)} paired seeds")
    print(f"dense control : {np.mean(d_vals):.4f} ± {sd(d_vals):.4f}")
    print(f"MoE recommended: {np.mean(m_vals):.4f} ± {sd(m_vals):.4f}")
    print(f"paired gap     : {np.mean(gaps):+.4f} ± {sd(gaps):.4f}   (min {min(gaps):+.4f}, max {max(gaps):+.4f})")
    print("all gaps negative:", all(g < 0 for g in gaps))
