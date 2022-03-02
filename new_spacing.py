import numpy as np

import SimpleITK as sitk

# path = '/media/qmia/tmdhey/LUNA/nii_data/SS0/1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.nii'
#
# image = sitk.ReadImage(path)
#
# resample = sitk.ResampleImageFilter()
# resample.SetInterpolator = sitk.sitkLinear
# resample.SetOutputDirection = image.GetDirection()
# resample.SetOutputOrigin = image.GetOrigin()
# orig_spacing = image.GetSpacing()
# # 정보 추가
# # x,y축을 건들이는 코드가 없음
# new_spacing = (orig_spacing[0], orig_spacing[1], 5)
# resample.SetOutputSpacing = new_spacing
#
# orig_size = np.array(image.GetSize(), dtype=np.int)
# # new_size = (orig_size)*((orig_spacing)/(new_spacing)) # tuple / tuple xxx
# new_size = np.array(orig_size)*(np.array(orig_spacing)/np.array(new_spacing))
# new_size= tuple(new_size)
# new_size = np.ceil(new_size).astype(np.int)
# new_size = [int(s) for s in new_size]
# resample.SetSize = new_size
#
# newimage = resample.Execute(image)
# # normalization 적용
#
# print(image.GetSpacing())
# print(resample.SetOutputSpacing)
# print(newimage.GetSpacing())
# print(resample)
# # sitk.WriteImage(newimage,'/media/qmia/tmdhey/LUNA/new_spacing_image.nii')

# ------------------------------------------------------------------
path = '/media/qmia/tmdhey/LUNA/nii_data/SS0/1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.nii'

image = sitk.ReadImage(path)

resample = sitk.ResampleImageFilter()
resample.SetInterpolator = sitk.sitkLinear
resample.SetOutputDirection(image.GetDirection())
resample.SetOutputOrigin(image.GetOrigin())
orig_spacing = image.GetSpacing()
# 정보 추가
# x,y축을 건들이는 코드가 없음
new_spacing = (orig_spacing[0], orig_spacing[1], 5)
resample.SetOutputSpacing(new_spacing)

orig_size = np.array(image.GetSize(), dtype=np.int)
# new_size = (orig_size)*((orig_spacing)/(new_spacing)) # tuple / tuple xxx
new_size = np.array(orig_size)*(np.array(orig_spacing)/np.array(new_spacing))
new_size= tuple(new_size)
new_size = np.ceil(new_size).astype(np.int)
new_size = [int(s) for s in new_size]
resample.SetSize(new_size)

newimage = resample.Execute(image)
# normalization 적용

print(image.GetSpacing())
print(resample.SetOutputSpacing)
print(newimage.GetSpacing())
print(resample)
sitk.WriteImage(newimage,'/media/qmia/tmdhey/LUNA/new_spacing_image.nii')