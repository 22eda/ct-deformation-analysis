import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load(p):
    return list(csv.DictReader(open(p)))

m = load("evaluation_results/metrics_test.csv")
b = load("evaluation_results/baseline_test.csv")
structs = ["prostate", "bladder", "femur_heads"]

def stats(rows, st, met):
    v = [float(r[met]) for r in rows if r["structure"] == st]
    return np.mean(v), np.std(v), len(v)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
x = np.arange(3)
for ax, met in zip(axes[:2], ["gDice", "ADICE"]):
    mm = [stats(m, s, met) for s in structs]
    bb = [stats(b, s, met) for s in structs]
    ax.bar(x - 0.2, [t[0] for t in mm], 0.4, yerr=[t[1] for t in mm], label="Model", capsize=3)
    ax.bar(x + 0.2, [t[0] for t in bb], 0.4, yerr=[t[1] for t in bb], label="Baseline (deformasyon yok)", capsize=3)
    ax.set_xticks(x); ax.set_xticklabels(structs)
    ax.set_ylim(0, 1); ax.set_title(f"{met} (test, n={mm[0][2]} hasta)")
    ax.legend(fontsize=8)
    for s, a, c in zip(structs, mm, bb):
        print(f"{met:6s} {s:12s} model={a[0]:.3f}  baseline={c[0]:.3f}  (n={a[2]}/{c[2]})")

tgt = [1.374, 1.437, 1.284]
prd = [0.473, 0.573, 0.437]
axes[2].bar(np.arange(3) - 0.2, tgt, 0.4, label="Gercek (hedef)")
axes[2].bar(np.arange(3) + 0.2, prd, 0.4, label="Model tahmini")
axes[2].set_xticks(np.arange(3)); axes[2].set_xticklabels(["Patient_05", "Patient_12", "Patient_14"])
axes[2].set_ylabel("ort. hareket (mm)")
axes[2].set_title("Deformasyon buyuklugu\n(ilk model, tum hastalarla egitilmis)")
axes[2].legend(fontsize=8)
plt.tight_layout()
plt.savefig("figures/summary_model_vs_baseline.png", dpi=150)
print("Kaydedildi: figures/summary_model_vs_baseline.png")
