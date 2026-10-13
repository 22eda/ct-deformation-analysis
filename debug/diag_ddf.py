import os, glob
import numpy as np, torch, nibabel as nib

cache = os.path.join(os.environ["SCRATCH"], "deformation_project", "cache")
print("== 1) Gercek (hedef) deformasyon alanlari ==")
for pid in ["Patient_01", "Patient_05", "Patient_30"]:
    for j in [5, 20]:
        f = os.path.join(cache, f"{pid}_frac{j}.pt")
        if not os.path.exists(f):
            continue
        d = torch.load(f)["def"].numpy()          # [3,H,W,D], mm
        mag = np.sqrt((d ** 2).sum(0))
        body = mag > 0
        print(f"{pid} frac{j}: sifir-tahmin MSE={np.mean(d**2):.4f} | "
              f"vucut ici ort buyukluk={mag[body].mean():.3f} mm, "
              f"%99={np.percentile(mag[body],99):.2f} mm, max={mag.max():.2f} mm")
    
print("\n== 2) Uretilen varyantlar planning'den ne kadar farkli? ==")
STRUCT = "/net/tscratch/people/plgztabor/ROBUST_PLANNING/DATA/STRUCTURES"
for pid in ["Patient_05", "Patient_12"]:
    base = np.asarray(nib.load(f"{STRUCT}/{pid}/{pid}_fraction_1_.nii.gz").dataobj).squeeze()
    for k in [0, 1]:
        v = np.load(f"generated_variants/{pid}/variant_seg_{k:02d}.npy", mmap_mode="r")
        b = base[:v.shape[0], :v.shape[1], :v.shape[2]]
        diff = np.mean(np.asarray(v) != b)
        print(f"{pid} varyant {k}: planning'den farkli voksel orani = {diff*100:.3f} %")
