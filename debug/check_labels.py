import nibabel as nib
import numpy as np

img = nib.load("/net/tscratch/people/plgztabor/ROBUST_PLANNING/DATA/STRUCTURES/Patient_01/Patient_01_fraction_1_.nii.gz")
data = img.get_fdata()
zooms = img.header.get_zooms()[:3]  # (dx, dy, dz) mm
voxel_vol_cc = (zooms[0] * zooms[1] * zooms[2]) / 1000.0  # mm^3 -> cc

print(f"Voxel boyutu: {zooms} mm, voxel hacmi: {voxel_vol_cc:.5f} cc")
print(f"Image shape: {data.shape}\n")

for v in np.unique(data):
    if v == 0:
        continue
    mask = (data == v)
    n_vox = int(mask.sum())
    volume_cc = n_vox * voxel_vol_cc
    coords = np.argwhere(mask)
    centroid = coords.mean(axis=0)  # [x,y,z] indeks ortalamasi
    x_range = (coords[:,0].min(), coords[:,0].max())
    print(f"Label {int(v)}: {n_vox} voksel, hacim={volume_cc:.1f} cc, "
          f"centroid(idx)={centroid}, x_range={x_range}")
