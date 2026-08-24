import napari
import SimpleITK as sitk

# Doğrudan C sürücüsündeki klasör yolları
def_field_path = r"C:\Patient_01\deformation_fields\def_field_frac1_to_frac2.nii.gz"
frac1_path     = r"C:\Patient_01\Patient_01_fraction_1_.nii.gz"
frac2_path     = r"C:\Patient_01\Patient_01_fraction_2_.nii.gz"

print("Dosyalar yükleniyor...")
frac1_img = sitk.GetArrayFromImage(sitk.ReadImage(frac1_path))
frac2_img = sitk.GetArrayFromImage(sitk.ReadImage(frac2_path))
def_field = sitk.GetArrayFromImage(sitk.ReadImage(def_field_path))

print("Napari arayüzü açılıyor...")

# Napari viewer başlat
viewer = napari.Viewer()
viewer.add_image(frac1_img, name="Fraction 1 (Fixed)")
viewer.add_image(frac2_img, name="Fraction 2 (Moving)")
viewer.add_vectors(def_field, name="Deformation Vectors")

napari.run()

