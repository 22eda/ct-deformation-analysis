import nibabel as nib
import numpy as np

ct_img = nib.load("/net/tscratch/people/plgztabor/ROBUST_PLANNING/DATA/CT/Patient_05/Patient_05_fraction_1_.nii.gz")
seg_img = nib.load("/net/tscratch/people/plgztabor/ROBUST_PLANNING/DATA/STRUCTURES/Patient_05/Patient_05_fraction_1_.nii.gz")

print("CT shape:", ct_img.shape)
print("CT affine:\n", ct_img.affine)
print()
print("SEG shape:", seg_img.shape)
print("SEG affine:\n", seg_img.affine)
print()
print("Affine farki (CT - SEG):\n", ct_img.affine - seg_img.affine)
