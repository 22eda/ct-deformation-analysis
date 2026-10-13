# CT Deformation Analysis Pipeline

Research code, work in progress. Patient data is NOT included in this repository.

## Project Overview
Accurate anatomical alignment and deformation tracking are critical in **Robust Radiotherapy Planning**. Patient anatomy changes between treatment fractions due to factors like weight fluctuations, organ motion, and tumor shrinkage.

This repository contains code to (1) compute inter-fractional deformation fields with Demons registration (Fraction 1 as the reference against all later fractions), (2) train a 3D U-Net with Monte Carlo dropout to predict them, (3) generate anatomical variants from the planning CT, and (4) evaluate the variants against the real fractions.

## Pipeline and Repository Structure

| Step | Files | Purpose |
|---|---|---|
| 1. Deformation targets | `calculate_deformation.py`, `run_deformation.slurm` | Demons registration (SimpleITK), fraction 1 -> fraction n, body-masked displacement fields. Current settings: 15 iterations, single scale. |
| 2. Visual check | `visualize_result.py` | Interactive 3D inspection of the fields with napari (vector layers over the CT). |
| 3. Patient split | `make_split.py`, `patient_split.json` | Patient-level split, 39 train / 10 test, seed 42. |
| 4. Training | `dataset.py`, `model.py`, `train.py`, `train.slurm` | 3D U-Net, dropout 0.2, MSE loss, mixed precision, 15 epochs, train patients only. |
| 5. Variant generation | `generate_variants.py`, `generate_all.slurm` | 10 MC-dropout passes on the planning CT, inversion of each predicted field, warping of the CT (linear) and the segmentation (nearest neighbour). |
| 6. Evaluation | `metrics_light.py`, `metrics_baseline.py`, `metrics.slurm` | gDice, ADICE, GED vs. real fractions. The baseline repeats the unchanged planning mask 10 times. |
| 7. Figures | `viz_probmaps.py`, `viz_summary.py` | Probability maps and model-vs-baseline summary. |
| Diagnostics | `debug/` | Scripts used to check labels, alignment and displacement magnitudes. |

## Visualization of the Demons Fields

<img width="722" height="574" alt="27" src="https://github.com/user-attachments/assets/43684d29-954c-41a6-b055-5afb5d19cfd9" />

*Figure: Inter-fractional deformation vectors visualized over patient CT slices using napari.*

## Results (10 held-out test patients, mean)

| Structure | gDice | ADICE | GED |
|---|---|---|---|
| Prostate | 0.867 | 0.810 | 0.346 |
| Bladder | 0.923 | 0.898 | 0.207 |
| Femoral heads | 0.879 | 0.807 | 0.314 |

A no-deformation baseline scores almost identically, and the generated variants differ from the planning segmentation in only about 0.02% of voxels. So the model does not yet generate meaningful anatomical variability.

## Known Limitations
- Predicted displacements are about 0.5 mm on average vs. about 1.4 mm in the targets (measured on an earlier model trained on all 49 patients; not yet repeated on the final train/test model).
- Demons targets use 15 iterations at a single scale.
- Single train/test split, no cross-validation, no two-phase dropout training.
- No rectum label in the data. The label mapping (1 = body, 2 and 3 = femoral heads, 4 = prostate, 5 = bladder) was inferred from volumes and positions and is not confirmed.
- No dosimetric validation.
- Paths and Slurm account/partition are specific to the Athena cluster.

## Next Steps
- Repeat the displacement-magnitude check on the final train/test model and find out why predictions are smaller than the targets.
- Recompute the targets with multiscale diffeomorphic Demons, retrain (including two-phase dropout), and compare against the no-deformation baseline.
- Confirm the label mapping and whether a rectum label exists.
- Cross-validation instead of a single split.
- Dosimetric validation (mean dose and dose uncertainty maps), if in scope.
- Longer term: alternative generative models (e.g. latent diffusion or flow matching).
