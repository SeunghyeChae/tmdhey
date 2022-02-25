import dicom2nifti
import os
from glob import glob

# 1개 변경
# path= '/media/qmia/tmdhey/LUNA/dcm_data/'
# path_one_patient= '/media/qmia/tmdhey/LUNA/dcm_data/SS1/1.3.6.1.4.1.14519.5.2.1.6279.6001.100684836163890911914061745866'
# path_out_data= '/media/qmia/tmdhey/LUNA/nii_data/'
#
# dicom2nifti.dicom_series_to_nifti(path_one_patient, path_out_data+'dddd')
# # dicom2nifti.dicom_series_to_nifti(path_one_patient, os.path.join(path_out_data,))

# real code
path= '/media/qmia/tmdhey/LUNA/dcm_data'
subsets= os.listdir(path)

for subset in (subsets):
    dcm_path= path + '/' + subset
    # dcm_directory = glob(os.path.join(dcm_path))
    patient_n= os.listdir(dcm_path)

    for patient in patient_n:
        path_one_patient= dcm_path + '/' + patient
        path_out_data= '/media/qmia/tmdhey/LUNA/nii_data' + '/' + subset + '/' + patient
        dicom2nifti.dicom_series_to_nifti(path_one_patient,path_out_data +'.nii')






# # dcm key check
# import pydicom
#
# dcm_file_path= '/media/qmia/tmdhey/LUNA/dcm_data/SS1/1.3.6.1.4.1.14519.5.2.1.6279.6001.100684836163890911914061745866/0.dcm'
# # print(dcm_file[0])
# dcm_file= pydicom.dcmread(dcm_file_path)
# # print(dcm_file)
#
# '''see all keys in dcm file'''
# for key in dcm_file.__dir__():
#     # print(key)
#     print(key, dcm_file.data_element(key))
#
# '''get dimension and thickness information from the dicom file'''
# '''dims'''
# print('*********************************')
# # print(len(dcm_files)) # 원랜 여기까지가 dim인데 내가 파일 하나가져와서ㅋㅋ
# print('ROWS : ',dcm_file.Rows)
# print('COLUMNS : ', dcm_file.Columns)
# '''spacing'''
# # print('SLICE THICKNESS : ',dcm_file.SliceThickness)
# print('PIXEL SPACING : ',dcm_file.PixelSpacing)
# # print(dcm_file.pixel_array)