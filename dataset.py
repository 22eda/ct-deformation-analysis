import os
import json
import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np

CACHE_DIR = os.path.join(os.environ.get("SCRATCH", "."), "deformation_project", "cache")

class MedicalDataset(Dataset):
    def __init__(self, data_base_dir, results_base_dir, target_fractions=range(2, 28),
                 split_file=None, split="all"):
        self.samples = []
        os.makedirs(CACHE_DIR, exist_ok=True)

        allowed_patients = None
        if split_file is not None and split != "all":
            with open(split_file) as f:
                split_data = json.load(f)
            allowed_patients = set(split_data[split])

        if os.path.exists(results_base_dir):
            patient_ids = sorted([
                d for d in os.listdir(results_base_dir)
                if os.path.isdir(os.path.join(results_base_dir, d)) and d.startswith("Patient_")
            ])
        else:
            patient_ids = []

        if allowed_patients is not None:
            patient_ids = [p for p in patient_ids if p in allowed_patients]

        print(f"Dataset taramasi baslatildi (split={split}). Bulunan hasta klasor sayisi: {len(patient_ids)}")

        for patient_id in patient_ids:
            patient_res_dir = os.path.join(results_base_dir, patient_id)
            patient_data_dir = os.path.join(data_base_dir, patient_id)
            def_fields_dir = os.path.join(patient_res_dir, "deformation_fields")

            for j in target_fractions:
                moving_path_1 = os.path.join(patient_res_dir, f"{patient_id}_fraction_{j}_.nii.gz")
                moving_path_2 = os.path.join(patient_data_dir, f"{patient_id}_fraction_{j}_.nii.gz")

                moving_path = None
                if os.path.exists(moving_path_1):
                    moving_path = moving_path_1
                elif os.path.exists(moving_path_2):
                    moving_path = moving_path_2

                def_field_path = os.path.join(def_fields_dir, f"def_field_frac1_to_frac{j}.nii.gz")

                if moving_path and os.path.exists(def_field_path):
                    self.samples.append({
                        "patient": patient_id,
                        "fraction": j,
                        "moving": moving_path,
                        "def_field": def_field_path,
                        "cache": os.path.join(CACHE_DIR, f"{patient_id}_frac{j}.pt"),
                    })

        print(f"Basariyla yuklenen gecerli ornek sayisi (num_samples): {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _normalize(x):
        return torch.clamp(x, -1000.0, 1000.0) / 1000.0

    def _load_and_cache(self, item):
        moving_img = np.squeeze(nib.load(item["moving"]).get_fdata(dtype=np.float32))
        def_field_img = np.squeeze(nib.load(item["def_field"]).get_fdata(dtype=np.float32))

        d1 = (moving_img.shape[0] // 16) * 16
        d2 = (moving_img.shape[1] // 16) * 16
        d3 = (moving_img.shape[2] // 16) * 16

        moving_img = np.ascontiguousarray(moving_img[:d1, :d2, :d3])
        def_field_img = np.ascontiguousarray(def_field_img[:d1, :d2, :d3, :])

        moving_tensor = torch.from_numpy(moving_img).unsqueeze(0)
        def_tensor = torch.from_numpy(def_field_img).permute(3, 0, 1, 2).contiguous()
        del moving_img, def_field_img

        tmp_path = item["cache"] + f".tmp{os.getpid()}"
        torch.save({"moving": moving_tensor, "def": def_tensor}, tmp_path)
        os.replace(tmp_path, item["cache"])
        return moving_tensor, def_tensor

    def __getitem__(self, idx):
        item = self.samples[idx]

        if os.path.exists(item["cache"]):
            try:
                data = torch.load(item["cache"])
                return self._normalize(data["moving"]), data["def"]
            except Exception as e:
                print(f"Bozuk cache, yeniden uretiliyor: {item['cache']} ({e})", flush=True)

        moving_tensor, def_tensor = self._load_and_cache(item)
        return self._normalize(moving_tensor), def_tensor
