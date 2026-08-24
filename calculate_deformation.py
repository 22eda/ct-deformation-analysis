import SimpleITK as sitk
import os
import glob

def calculate_deformation_for_all_patients(data_base_dir, output_base_dir):
    """
    Iterates through all patient directories, takes fraction 1 as the fixed reference image,
    and calculates dense deformation fields for all subsequent fractions (J > 1).
    """
    # Find all patient folders matching the pattern Patient_*
    patient_dirs = sorted(glob.glob(os.path.join(data_base_dir, "Patient_*")))
    
    if not patient_dirs:
        print("No patient directories found!")
        return

    print(f"Found {len(patient_dirs)} patients. Starting processing...\n")

    for patient_path in patient_dirs:
        patient_id = os.path.basename(patient_path)
        fixed_path = os.path.join(patient_path, f"{patient_id}_fraction_1_.nii.gz")
        
        # Skip if the baseline image (fraction 1) doesn't exist
        if not os.path.exists(fixed_path):
            print(f"Warning: {fixed_path} not found, skipping.")
            continue

        print(f"--- Processing: {patient_id} ---")
        fixed_image = sitk.ReadImage(fixed_path, sitk.sitkFloat32)

        # Dynamically determine the total number of fractions available for the patient
        fraction_files = glob.glob(os.path.join(patient_path, f"{patient_id}_fraction_*_.nii.gz"))
        max_fraction = len(fraction_files)

        # Compute deformation fields relative to fraction 1 for each subsequent fraction (J > 1)
        for j in range(2, max_fraction + 1):
            moving_path = os.path.join(patient_path, f"{patient_id}_fraction_{j}_.nii.gz")
            
            if os.path.exists(moving_path):
                moving_image = sitk.ReadImage(moving_path, sitk.sitkFloat32)
                print(f"  {patient_id}: fraction_1 -> fraction_{j} registering...")
                
                # Configure Demons Registration filter for dense deformation field estimation
                demons_filter = sitk.DemonsRegistrationFilter()
                demons_filter.SetNumberOfIterations(15)
                demons_filter.SetStandardDeviations(1.0)
                
                # Execute registration to extract the displacement field
                displacement_field = demons_filter.Execute(fixed_image, moving_image)
                
                # Save the resulting deformation field to the personal scratch results directory
                output_dir = os.path.join(output_base_dir, patient_id, "deformation_fields")
                os.makedirs(output_dir, exist_ok=True)
                
                output_path = os.path.join(output_dir, f"def_field_frac1_to_frac{j}.nii.gz")
                sitk.WriteImage(displacement_field, output_path)
                print(f"  Saved: {output_path}")

# Define data paths (Read-only raw data from professor's path, outputs written to personal scratch)
data_base_path = "/net/tscratch/people/plgztabor/ROBUST_PLANNING/DATA/CT"
my_output_path = "/net/tscratch/people/plgedademirel13/deformation_project/results"

calculate_deformation_for_all_patients(data_base_path, my_output_path)
