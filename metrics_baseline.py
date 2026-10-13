import os, sys, json, csv
import numpy as np
import nibabel as nib

STRUCT_BASE = "/net/tscratch/people/plgztabor/ROBUST_PLANNING/DATA/STRUCTURES"
GEN_ROOT = "generated_variants"
OUT = "evaluation_results"
os.makedirs(OUT, exist_ok=True)

STRUCTURES = {"prostate": [4], "bladder": [5], "femur_heads": [2, 3]}
THRESHOLDS = [round(0.1 * k, 1) for k in range(1, 11)]


def load_real(path):
    data = np.asarray(nib.load(path).dataobj)
    data = np.squeeze(data)
    s = tuple((d // 16) * 16 for d in data.shape[:3])
    return data[:s[0], :s[1], :s[2]].astype(np.uint8)


def load_gen(path):
    return np.load(path, mmap_mode="r").astype(np.uint8)


def gdice(a, b):
    den = a.sum() + b.sum()
    return 1.0 if den == 0 else float(2 * np.sqrt(a * b).sum() / den)


def adice(a, b):
    sc = []
    for t in THRESHOLDS:
        x, y = a >= t, b >= t
        den = x.sum() + y.sum()
        sc.append(1.0 if den == 0 else 2.0 * np.logical_and(x, y).sum() / den)
    return float(np.mean(sc))


def iou_d(x, y):
    u = np.logical_or(x, y).sum()
    return 0.0 if u == 0 else 1.0 - np.logical_and(x, y).sum() / u


def mean_pair(ms):
    v = [iou_d(ms[i], ms[j]) for i in range(len(ms)) for j in range(i + 1, len(ms))]
    return float(np.mean(v)) if v else 0.0


def ged(g, p):
    cross = np.mean([iou_d(x, y) for x in g for y in p])
    return float(2 * cross - mean_pair(g) - mean_pair(p))


def evaluate(pid):
    real = []
    for j in range(2, 28):
        f = os.path.join(STRUCT_BASE, pid, f"{pid}_fraction_{j}_.nii.gz")
        if os.path.exists(f):
            real.append(load_real(f))
    gen = []
    base = load_real(os.path.join(STRUCT_BASE, pid, f"{pid}_fraction_1_.nii.gz"))
    for k in range(10):
        gen.append(base.copy())
    if not real or len(gen) < 10:
        print(f"  {pid}: veri eksik (real={len(real)}, gen={len(gen)}), atlaniyor", flush=True)
        return []

    shapes = [s.shape for s in real + gen]
    cs = tuple(min(s[i] for s in shapes) for i in range(3))
    real = [s[:cs[0], :cs[1], :cs[2]] for s in real]
    gen = [s[:cs[0], :cs[1], :cs[2]] for s in gen]

    rows = []
    for name, labels in STRUCTURES.items():
        gm = [np.isin(s, labels) for s in real]
        pm = [np.isin(s, labels) for s in gen]
        union = np.logical_or.reduce(gm + pm)
        if not union.any():
            continue
        idx = np.argwhere(union)
        lo = np.maximum(idx.min(0) - 2, 0)
        hi = idx.max(0) + 3
        sl = tuple(slice(lo[i], hi[i]) for i in range(3))
        gm = [m[sl] for m in gm]
        pm = [m[sl] for m in pm]
        p_gt = np.mean(gm, axis=0, dtype=np.float32)
        p_pr = np.mean(pm, axis=0, dtype=np.float32)
        r = {"patient": pid, "structure": name, "n_real": len(real), "n_gen": len(gen),
             "gDice": gdice(p_gt, p_pr), "ADICE": adice(p_gt, p_pr), "GED": ged(gm, pm)}
        rows.append(r)
        print(f"  {pid}/{name}: gDice={r['gDice']:.3f} ADICE={r['ADICE']:.3f} GED={r['GED']:.3f}", flush=True)
    return rows


if __name__ == "__main__":
    split = json.load(open("patient_split.json"))
    which = sys.argv[1] if len(sys.argv) > 1 else "test"
    names = ["test", "train"] if which == "all" else [which]
    allrows = []
    for sn in names:
        print(f"=== {sn.upper()} ===", flush=True)
        for pid in split[sn]:
            for r in evaluate(pid):
                r["split"] = sn
                allrows.append(r)
            path = os.path.join(OUT, f"baseline_{which}.csv")
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["split", "patient", "structure", "n_real", "n_gen", "gDice", "ADICE", "GED"])
                w.writeheader()
                w.writerows(allrows)
    print("\n=== ORTALAMALAR ===")
    for sn in names:
        for st in STRUCTURES:
            sel = [r for r in allrows if r["split"] == sn and r["structure"] == st]
            if sel:
                print(f"{sn:5s} {st:12s} gDice={np.mean([r['gDice'] for r in sel]):.3f} "
                      f"ADICE={np.mean([r['ADICE'] for r in sel]):.3f} "
                      f"GED={np.mean([r['GED'] for r in sel]):.3f} (n={len(sel)})")
    print("BITTI")
