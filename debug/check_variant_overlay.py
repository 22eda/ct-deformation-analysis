import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

patient = "Patient_05"
variant_ct = np.load(f"generated_variants/{patient}/variant_ct_00.npy")
variant_seg = np.load(f"generated_variants/{patient}/variant_seg_00.npy")

mid = variant_ct.shape[2] // 2

fig, ax = plt.subplots(1, 1, figsize=(7, 7))
ax.imshow(variant_ct[:, :, mid].T, cmap="gray", origin="lower", vmin=-200, vmax=200)
body = (variant_seg[:, :, mid] == 1).astype(float)
ax.contour(body.T, levels=[0.5], colors="red", linewidths=2, origin="lower")
bladder = (variant_seg[:, :, mid] == 5).astype(float)
ax.contour(bladder.T, levels=[0.5], colors="cyan", linewidths=2, origin="lower")
prostate = (variant_seg[:, :, mid] == 4).astype(float)
ax.contour(prostate.T, levels=[0.5], colors="yellow", linewidths=2, origin="lower")
ax.set_title(f"{patient} - Uretilen varyant (kirmizi=vucut, cyan=mesane, sari=prostat)")
ax.axis("off")

plt.tight_layout()
plt.savefig("figures/variant_overlay.png", dpi=150)
print("Kaydedildi: figures/variant_overlay.png")
