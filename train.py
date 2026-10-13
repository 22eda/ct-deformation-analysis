import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from model import Probabilistic3DUNet
from dataset import MedicalDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

data_base_dir = "/net/tscratch/people/plgztabor/ROBUST_PLANNING/DATA/CT"
results_base_dir = "/net/tscratch/people/plgedademirel13/deformation_project/results"

dataset = MedicalDataset(
    data_base_dir=data_base_dir,
    results_base_dir=results_base_dir,
    target_fractions=range(2, 28),
    split_file="patient_split.json",
    split="train",   # <-- SADECE train hastalari
)

if len(dataset) == 0:
    raise ValueError("Egitim icin gecerli veri bulunamadi! Yollari kontrol edin.")

dataloader = DataLoader(
    dataset, batch_size=1, shuffle=True,
    num_workers=4, pin_memory=True, persistent_workers=True,
)

model = Probabilistic3DUNet(in_channels=1, out_channels=3, dropout_prob=0.2).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scaler = GradScaler()

num_epochs = 15
print(f"TRAIN split ile model egitimi baslatiliyor ({num_epochs} Epoch, {len(dataset)} ornek)...")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for step, (moving_tensor, def_tensor) in enumerate(dataloader):
        moving_tensor = moving_tensor.to(device, non_blocking=True)
        def_tensor = def_tensor.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            predicted_def = model(moving_tensor)
            loss = criterion(predicted_def, def_tensor)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Ortalama Loss: {epoch_loss:.6f}")
    torch.save(model.state_dict(), f"checkpoint_epoch_{epoch+1}.pth")

torch.save(model.state_dict(), "probabilistic_unet_model_TRAINSPLIT.pth")
print("Egitim basariyla tamamlandi! Model 'probabilistic_unet_model_TRAINSPLIT.pth' olarak kaydedildi.")
