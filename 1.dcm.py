from glob import glob
import SimpleITK as sitk
import sys
import time
import os
from pathlib import Path

# 함수
def writeSlices(series_tag_values, new_img, out_dir, i):
    image_slice = new_img[:, :, i]

    # tag shared by the series
    list(map(lambda tag_value: image_slice.SetMetaData(tag_value[0],
                                                       tag_value[1]),
         series_tag_values))

    image_slice.SetMetaData("0008|001", time.strftime("%Y%m%d"))  # instance creation date
    image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S")) #  time
    image_slice.SetMetaData("0008|0060", "CT") # thickness

    # Patient
    image_slice.SetMetaData("0020|0032", '\\'.join(
        map(str, new_img.TransformIndexToPhysicalPoint((0, 0, i)))))
    # Instance Number
    image_slice.SetMetaData("0020,0013", str(i))

    # write dicom format
    writer.SetFileName(os.path.join(out_dir, str(i) + '.dcm'))
    writer.Execute(image_slice)

# ---------------------------------------------------------------------------
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# 폴더 생성
out_path= '/media/qmia/tmdhey/LUNA/dcm_data'
path= '/media/qmia/tmdhey/LUNA/raw_data'
subsets= os.listdir(path)

for subset in (subsets):
    mhd_path= path + '/' + subset
    mhd_files= glob(os.path.join(mhd_path, '*.mhd'))

    for mhd_file in mhd_files:
        file_name = Path(mhd_file).stem
        createFolder(out_path + '/' + subset + '/' + file_name)

# ----------------------------------------------------------------------------
# dcm생성
for subset in (subsets):
    mhd_path= path + '/' + subset
    mhd_files= glob(os.path.join(mhd_path, '*.mhd'))

    for mhd_file in mhd_files:
        ###### file_name/ dcm_path
        file_name= Path(mhd_file).stem
        out_dcm_path= out_path + '/' + subset + '/' + file_name + '/'
        ######
        org= sitk.ReadImage(mhd_file)
        array= sitk.GetArrayFromImage(org)
        new_img = sitk.GetImageFromArray(array)
        new_img.SetSpacing(org.GetSpacing())
        # new_img = sitk.GetImageFromArray(*****)
        # new_img.SetSpacing([******)

        writer= sitk.ImageFileWriter()
        writer.KeepOriginalImageUIDOn()

        modification_time = time.strftime("%H%M%S")
        modification_date = time.strftime("%Y%m%d")


        # direction = new_img.GetDirection()
        direction = org.GetDirection()

        series_tag_values = [
            ("0008|0031", modification_time),  # Series Time
            ("0008|0021", modification_date),  # Series Date
            ("0008|0008", "DERIVED\\SECONDARY"),  # Image Type
            ("0020|000e", "1.2.826.0.1.3680043.2.1125."
             + modification_date + ".1" + modification_time),  # Series Instance UID
            ("0020|0037", '\\'.join(map(str, (direction[0], direction[3], direction[6],
                                              direction[1], direction[4],
                                              direction[7])))),
            # (Patient)
            ("0008|103e", "Created-SimpleITK")  # Series Description
        ]

        for i in range(array.shape[0]):
            writeSlices(series_tag_values, new_img, out_dcm_path, i)

