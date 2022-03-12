from tqdm import tqdm
import nibabel as nib
import os
from glob import glob
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import SimpleITK as sitk
import numpy as np
from pathlib import Path
import time

# --------------------------------------- ------------- --------------------------------------------
# --------------------------------------- defs -----------------------------------------------------
def augmentation(image,label):
    i = random.randint(0, 200)
    n = random.randint(1,3)
    image = tf.expand_dims(image, axis=-1) # 3 dimensional
    image = tf.image.random_flip_left_right(image, seed=i)
    image = tf.image.random_flip_up_down(image, seed=i)
    image = tf.image.rot90(image, k=n)

    return image,label

# --------------------------------------- ------------- --------------------------------------------
# --------------------------------------- train_test set -------------------------------------------
SS_path = '/media/qmia/tmdhey/LUNA/nii_data/'
SSs_path = glob(os.path.join(SS_path, "*"))

train_paths = []
test_paths = []
for ss_path in SSs_path:
    if 'SS4' in ss_path:
        ss_nii_paths = glob(os.path.join(ss_path, "*.nii"))
        for nii_path in ss_nii_paths:
            test_paths.append(nii_path)
    else:
        ss_nii_paths = glob(os.path.join(ss_path, "*.nii"))
        for nii_path in ss_nii_paths:
            train_paths.append(nii_path)


# print(len(train_paths))
# print(len(test_paths))


# --------------------------------------- ------------- --------------------------------------------
# --------------------------------------- train_set --------------------------------------------
train_x = []
train_y= []

t0 = time.process_time()
for path in train_paths:
    file_name = Path(path).stem
    # nimg = nib.load(path)
    # img = nib.load(path).get_fdata()
    img = sitk.ReadImage(path)


    img_array = np.array(sitk.GetArrayFromImage(img), dtype=np.int)

    ##### spacing / z축 slice 개수 맞추기
    num_slice = img_array.shape[0]
    x = num_slice / 150
    resample = sitk.ResampleImageFilter()  # 인수를 사용하지 않고 기본 매개변수를 초기화하는 기본 생성자
    resample.SetInterpolator = sitk.sitkLinear
    resample.SetOutputDirection(img.GetDirection())
    resample.SetOutputOrigin(img.GetOrigin())
    orig_spacing = img.GetSpacing()

    new_spacing = (orig_spacing[0], orig_spacing[1], orig_spacing[2]*x)
    resample.SetOutputSpacing(new_spacing)

    orig_size = np.array(img.GetSize(), dtype=np.int)
    new_size = np.array(orig_size) * (np.array(orig_spacing) / np.array(new_spacing))
    new_size = tuple(new_size)
    new_size = np.ceil(new_size).astype(np.int)
    new_size = [int(s) for s in new_size]
    resample.SetSize(new_size)

    newimage = resample.Execute(img)

    img = sitk.GetArrayFromImage(newimage) # 사용할 img array
    # print('Array : ', ary)
    print('spacing : ', newimage.GetSpacing())
    print('shape : ', img.shape)

    ##### normalize
    max_value = abs(img).max()
    img = img / max_value

    # label 만들기...
    num_list = [i for i in range(img.shape[0])]
    num_list.sort()

    class_num = 0

    try:
        for num in num_list:
            slice = img[num,:, :]
            slice = np.resize(slice, (512, 512))
            slice = slice.reshape(512, 512)
            class_num += 1
            train_x.append(slice)
            train_y.append(class_num)
            print('[성공] : ', file_name, class_num)

            # augmented images / label
            for i in range(2):
                aug_image, augimg_label = augmentation(slice, class_num)
                train_x.append(aug_image)
                train_y.append(augimg_label)
                print(f'(auged images_{i+1}) [성공] : ', file_name, class_num)

    except Exception as e:
        print('********[ERROR]********', e)



t1 = time.process_time()
print("TRAIN_Elapsed time:", t1 - t0)

train_x = np.array(train_x)
train_y = np.array(train_y)

print(len(train_x))
print(len(train_y))



def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
#
#
def make_example(img, label):
    img = img.astype(np.float32).tobytes()
    label = label.tobytes()
    feature = {'image':_bytes_feature(img),
               'label':_bytes_feature(label)}
    proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return proto.SerializeToString()

output_path_train = '/media/qmia/tmdhey/LUNA/__tfrecord__/train_0311.tfrecord'

with tf.io.TFRecordWriter(output_path_train) as writer:
    for img, label in zip(train_x, train_y):
        example = make_example(img, label)
        writer.write(example)



# --------------------------------------- ------------- --------------------------------------------
# --------------------------------------- test_set --------------------------------------------
test_x = []
test_y= []

t0 = time.process_time()
for path in test_paths:
    file_name = Path(path).stem
    # nimg = nib.load(path)
    # img = nib.load(path).get_fdata()
    img = sitk.ReadImage(path)


    img_array = np.array(sitk.GetArrayFromImage(img), dtype=np.int)

    ##### spacing / z축 slice 개수 맞추기
    num_slice = img_array.shape[0]
    x = num_slice / 150
    resample = sitk.ResampleImageFilter()  # 인수를 사용하지 않고 기본 매개변수를 초기화하는 기본 생성자
    resample.SetInterpolator = sitk.sitkLinear
    resample.SetOutputDirection(img.GetDirection())
    resample.SetOutputOrigin(img.GetOrigin())
    orig_spacing = img.GetSpacing()

    new_spacing = (orig_spacing[0], orig_spacing[1], orig_spacing[2]*x)
    resample.SetOutputSpacing(new_spacing)

    orig_size = np.array(img.GetSize(), dtype=np.int)
    new_size = np.array(orig_size) * (np.array(orig_spacing) / np.array(new_spacing))
    new_size = tuple(new_size)
    new_size = np.ceil(new_size).astype(np.int)
    new_size = [int(s) for s in new_size]
    resample.SetSize(new_size)

    newimage = resample.Execute(img)

    img = sitk.GetArrayFromImage(newimage) # 사용할 img array
    # print('Array : ', ary)
    print('spacing : ', newimage.GetSpacing())
    print('shape : ', img.shape)

    ##### normalize
    max_value = abs(img).max()
    img = img / max_value

    # label 만들기...
    num_list = [i for i in range(img.shape[0])]
    num_list.sort()

    class_num = 0
    for num in num_list:
        slice = img[num,:, :]
        slice = np.resize(slice, (512, 512))
        slice = slice.reshape(512, 512)
        class_num += 1
        test_x.append(slice)
        test_y.append(class_num)
        print('[성공] : ', file_name , class_num)

t1 = time.process_time()
print("TEST_Elapsed time:", t1 - t0)

test_x = np.array(test_x)
test_y = np.array(test_y)

print(len(test_x))
print(len(test_y))

output_path_test= '/media/qmia/tmdhey/LUNA/__tfrecord__/test_0311.tfrecord'

with tf.io.TFRecordWriter(output_path_test) as writer:
    for img, label in zip(test_x, test_y):
        example = make_example(img, label)
        writer.write(example)


