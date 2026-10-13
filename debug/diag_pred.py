import os, glob, re
import numpy as np, torch

cache = os.path.join(os.environ["SCRATCH"], "deformation_project", "cache")
for pid in ["Patient_05", "Patient_12", "Patient_14"]:
    mf = glob.glob(f"inference_results_all/{pid}/mean_deformation_frac*.npy")
    if not mf:
        print(pid, "dosya yok"); continue
    frac = int(re.search(r"frac(\d+)", mf[0]).group(1))
    pred = np.load(mf[0])
    unc = np.load(f"inference_results_all/{pid}/uncertainty_frac{frac}.npy")
    tgt = torch.load(os.path.join(cache, f"{pid}_frac{frac}.pt"))["def"].numpy()
    body = np.sqrt((tgt ** 2).sum(0)) > 0
    pm = np.sqrt((pred ** 2).sum(0))
    tm = np.sqrt((tgt ** 2).sum(0))
    print(f"{pid} frac{frac}:")
    print(f"  hedef  ort buyukluk (vucut ici): {tm[body].mean():.3f}")
    print(f"  tahmin ort buyukluk (vucut ici): {pm[body].mean():.3f}")
    print(f"  tahminler arasi std (ort)      : {np.sqrt((unc**2).sum(0))[body].mean():.3f}")
    print(f"  MSE tahmin-hedef: {np.mean((pred-tgt)**2):.4f} | sifir-tahmin MSE: {np.mean(tgt**2):.4f}")
