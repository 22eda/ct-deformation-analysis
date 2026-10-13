import os
import json
import random

random.seed(42)  # tekrarlanabilirlik icin sabit seed

results_base_dir = "/net/tscratch/people/plgedademirel13/deformation_project/results"
patient_ids = sorted([
    d for d in os.listdir(results_base_dir)
    if os.path.isdir(os.path.join(results_base_dir, d)) and d.startswith("Patient_")
])

print(f"Toplam hasta sayisi: {len(patient_ids)}")

shuffled = patient_ids.copy()
random.shuffle(shuffled)

n_test = max(1, round(len(shuffled) * 0.2))  # ~%20 test
test_patients = sorted(shuffled[:n_test])
train_patients = sorted(shuffled[n_test:])

split = {"train": train_patients, "test": test_patients}

with open("patient_split.json", "w") as f:
    json.dump(split, f, indent=2)

print(f"Train hasta sayisi: {len(train_patients)}")
print(f"Test hasta sayisi: {len(test_patients)}")
print(f"Test hastalari: {test_patients}")
print("Split 'patient_split.json' dosyasina kaydedildi.")
