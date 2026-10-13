import sys
import numpy as np, nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STRUCT = "/net/tscratch/people/plgztabor/ROBUST_PLANNING/DATA/STRUCTURES"
pid = sys.argv[1] if len(sys.argv) > 1 else "Patient_05"
STRUCTURES = {"Prostate": [4], "Bladder": [5], "Femoral heads": [2, 3]}


def load_real(path):
    d = np.squeeze(np.asarray(nib.load(path).dataobj))
    s = tuple((n // 16) * 16 for n in d.shape[:3])
    return d[:s[0], :s[1], :s[2]].astype(np.uint8)


plan = load_real(f"{STRUCT}/{pid}/{pid}_fraction_1_.nii.gz")
shape = plan.shape
sum_real = {k: np.zeros(shape, np.float32) for k in STRUCTURES}
sum_gen = {k: np.zeros(shape, np.float32) for k in STRUCTURES}
n_real = n_gen = 0


def fit(a):
    a = a[:shape[0], :shape[1], :shape[2]]
    return a if a.shape == shape else None


for j in range(2, 28):
    try:
        a = fit(load_real(f"{STRUCT}/{pid}/{pid}_fraction_{j}_.nii.gz"))
    except FileNotFoundError:
        continue
    if a is None:
        print("shape mismatch, skipped fraction", j)
        continue
    for k, labs in STRUCTURES.items():
        sum_real[k] += np.isin(a, labs)
    n_real += 1

for i in range(10):
    a = fit(np.asarray(np.load(f"generated_variants/{pid}/variant_seg_{i:02d}.npy", mmap_mode="r")))
    if a is None:
        continue
    for k, labs in STRUCTURES.items():
        sum_gen[k] += np.isin(a, labs)
    n_gen += 1

print(f"{pid}: real fractions={n_real}, generated variants={n_gen}")
fig, axes = plt.subplots(3, 3, figsize=(11, 10))
cols = ["Real fractions (mean of masks)", "Model variants (mean of 10)", "Baseline (planning mask)"]
for r, (name, labs) in enumerate(STRUCTURES.items()):
    p_gt = sum_real[name] / max(n_real, 1)
    p_md = sum_gen[name] / max(n_gen, 1)
    p_bs = np.isin(plan, labs).astype(np.float32)
    union = (p_gt > 0) | (p_md > 0) | (p_bs > 0)
    if not union.any():
        print(name, ": EMPTY, no voxels with these labels")
        continue
    idx = np.argwhere(union)
    lo = np.maximum(idx.min(0) - 8, 0)
    hi = idx.max(0) + 9
    z = int(np.argmax(p_gt.sum(axis=(0, 1))))
    print(f"{name}: slice z={z}, GT nonzero px in slice={(p_gt[:, :, z] > 0).sum()}, "
          f"model nonzero={(p_md[:, :, z] > 0).sum()}, baseline nonzero={(p_bs[:, :, z] > 0).sum()}, "
          f"crop x={lo[0]}:{hi[0]} y={lo[1]}:{hi[1]}")
    for c, arr in enumerate([p_gt, p_md, p_bs]):
        ax = axes[r, c]
        im = ax.imshow(arr[lo[0]:hi[0], lo[1]:hi[1], z].T, cmap="gray", vmin=0, vmax=1, origin="lower")
        ax.axis("off")
        if r == 0:
            ax.set_title(cols[c], fontsize=10)
        if c == 0:
            ax.text(-0.04, 0.5, f"{name}\n(slice z={z})", transform=ax.transAxes,
                    rotation=90, va="center", ha="right", fontsize=10)
fig.subplots_adjust(left=0.08, right=0.88, top=0.92, bottom=0.03, wspace=0.05, hspace=0.08)
cax = fig.add_axes([0.9, 0.3, 0.015, 0.4])
fig.colorbar(im, cax=cax, label="Probability")
fig.suptitle(f"{pid} (test set): spatial probability maps", fontsize=12)
out = f"figures/probmaps_{pid}_en.png"
plt.savefig(out, dpi=150)
print("Saved:", out)
