import os
import sys
import json
import torch
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from model import Probabilistic3DUNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "probabilistic_unet_model_TRAINSPLIT.pth"
SPLIT_FILE = "patient_split.json"
DATA_BASE = "/net/tscratch/people/plgztabor/ROBUST_PLANNING/DATA/CT"
STRUCT_BASE = "/net/tscratch/people/plgztabor/ROBUST_PLANNING/DATA/STRUCTURES"
OUT_ROOT = "generated_variants"
NUM_MC = 10
BODY_LABEL = 1  # skin/body

os.makedirs(OUT_ROOT, exist_ok=True)

model = Probabilistic3DUNet(in_channels=1, out_channels=3, dropout_prob=0.2).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.train()

with open(SPLIT_FILE) as f:
    split = json.load(f)

def load_and_crop(path):
    img = nib.load(path)
    data = np.squeeze(img.get_fdata(dtype=np.float32))
    d1 = (data.shape[0] // 16) * 16
    d2 = (data.shape[1] // 16) * 16
    d3 = (data.shape[2] // 16) * 16
    data = data[:d1, :d2, :d3]
    spacing = img.header.get_zooms()[:3]
    return data, spacing

def numpy_field_to_sitk(field_np, spacing):
    field_zyx = np.transpose(field_np, (3, 2, 1, 0))
    sitk_field = sitk.GetImageFromArray(field_zyx.astype(np.float64), isVector=True)
    sitk_field.SetSpacing([float(s) for s in spacing])
    return sitk_field

def process_patient(patient_id, split_name):
    planning_ct_path = os.path.join(DATA_BASE, patient_id, f"{patient_id}_fraction_1_.nii.gz")
    planning_seg_path = os.path.join(STRUCT_BASE, patient_id, f"{patient_id}_fraction_1_.nii.gz")

    if not (os.path.exists(planning_ct_path) and os.path.exists(planning_seg_path)):
        print(f"Atlaniyor (dosya eksik): {patient_id}")
        return False

    last_file = os.path.join(OUT_ROOT, patient_id, f"variant_seg_{NUM_MC-1:02d}.npy")
    if os.path.exists(last_file):
        print(f"[{patient_id}] zaten uretilmis, atlaniyor", flush=True)
        return True

    print(f"[{patient_id}] CT ve segmentasyon yukleniyor...", flush=True)
    ct_data, ct_spacing = load_and_crop(planning_ct_path)
    seg_data, _ = load_and_crop(planning_seg_path)
    print(f"[{patient_id}] CT shape={ct_data.shape}, spacing={ct_spacing}", flush=True)

    body_mask = (seg_data == BODY_LABEL).astype(np.float32)  # [x,y,z]

    moving_tensor = torch.from_numpy(ct_data).unsqueeze(0).unsqueeze(0).to(device)
    moving_norm = torch.clamp(moving_tensor, -1000.0, 1000.0) / 1000.0

    patient_out = os.path.join(OUT_ROOT, patient_id)
    os.makedirs(patient_out, exist_ok=True)

    spacing = tuple(float(s) for s in ct_spacing)

    planning_ct_sitk = sitk.GetImageFromArray(np.transpose(ct_data, (2, 1, 0)).astype(np.float32))
    planning_ct_sitk.SetSpacing(spacing)
    planning_seg_sitk = sitk.GetImageFromArray(np.transpose(seg_data, (2, 1, 0)).astype(np.float32))
    planning_seg_sitk.SetSpacing(spacing)

    with torch.no_grad():
        for k in range(NUM_MC):
            print(f"[{patient_id}] MC varyant {k+1}/{NUM_MC}...", flush=True)
            pred_field = model(moving_norm)
            field_np = pred_field.squeeze(0).cpu().numpy()  # [3,x,y,z]

            # KRITIK DUZELTME: vucut disini sifirla (egitim hedefleriyle ayni mantik)
            field_np = field_np * body_mask[None, :, :, :]

            sitk_field = numpy_field_to_sitk(field_np, spacing)
            inverter = sitk.InvertDisplacementFieldImageFilter()
            inverter.SetMaximumNumberOfIterations(50)
            inverted_field = inverter.Execute(sitk_field)
            transform = sitk.DisplacementFieldTransform(inverted_field)

            resampler_ct = sitk.ResampleImageFilter()
            resampler_ct.SetReferenceImage(planning_ct_sitk)
            resampler_ct.SetTransform(transform)
            resampler_ct.SetInterpolator(sitk.sitkLinear)
            resampler_ct.SetDefaultPixelValue(-1000)
            variant_ct = resampler_ct.Execute(planning_ct_sitk)

            resampler_seg = sitk.ResampleImageFilter()
            resampler_seg.SetReferenceImage(planning_seg_sitk)
            resampler_seg.SetTransform(transform)
            resampler_seg.SetInterpolator(sitk.sitkNearestNeighbor)
            resampler_seg.SetDefaultPixelValue(0)
            variant_seg = resampler_seg.Execute(planning_seg_sitk)

            variant_ct_np = np.transpose(sitk.GetArrayFromImage(variant_ct), (2, 1, 0))
            variant_seg_np = np.transpose(sitk.GetArrayFromImage(variant_seg), (2, 1, 0))

            np.save(os.path.join(patient_out, f"variant_ct_{k:02d}.npy"), variant_ct_np)
            np.save(os.path.join(patient_out, f"variant_seg_{k:02d}.npy"), variant_seg_np)

    print(f"[{split_name}] {patient_id}: {NUM_MC} varyant uretildi -> {patient_out}", flush=True)
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_patient = sys.argv[1]
        split_name = "test" if test_patient in split["test"] else "train"
        process_patient(test_patient, split_name)
    else:
        for split_name in ["test", "train"]:
            for patient_id in split[split_name]:
                process_patient(patient_id, split_name)

    print("Tamamlandi.")
