import tensorflow as tf
import pydicom as dcm
import pathlib
import os
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# np.set_printoptions(precision=4) # 소수점
#
# dcm_path= '/media/qmia/tmdhey/LUNA/dcm_data/SS0/1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260'
# # dcm_files= list_files(os.path.join(dcm_path,"*.dcm"))
# # print(dcm_files)
# list_ds= tf.data.Dataset.list_files(os.path.join(dcm_path,"*.dcm"),shuffle=False)
# print(list_ds)
# # dataset= tf.data.Dataset.from_tensor_slices()
#
# for elem in list_ds:
#   print(elem.numpy())

# dcm_path= '/media/qmia/tmdhey/LUNA/dcm_data/SS0/1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260'
#
# dcm_files= glob(os.path.join(dcm_path,"*.dcm"))

def encode2TfRecord():
    flag= 1
    imagepath= '/media/qmia/tmdhey/LUNA/dcm_data/SS0/1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260'
    if not os.path.exists(imagepath):
        print(imagepath + 'not exist')
        return
    allFulldose= glob(os.path.join(imagepath,"*.dcm"))

    def path2img(imgpaths):
        images= []
        for imgPath in imgpaths:
            image_bytes= dcm.dcmread(imgPath)

            image= image_bytes.pixel_array
            images.append(image)
        return images

    print(len(allFulldose))
    fulldose_imgs = path2img(allFulldose)
    fulldose_imgs = np.array(fulldose_imgs,dtype='int16')
    dataset = tf.data.Dataset.from_tensor_slices((fulldose_imgs))


    def _bytes_feature(value):
        '''Returns a bytes_list from a string / byte.'''
        if  isinstance(value, type(tf.constant(0))):
            value= value.numpy() # ByteList won't unpack a string from an EagerTensor
        return tf.train.Feature(bytes_list= tf.train.BytesList(value=[value]))

    def serialize_exemple(full_img):
        '''Creates a tf.Example message ready to be written to a file'''
        # Creates a dictionary message readyto be written to a file
        # data type

        full_img= full_img.numpy().tobytes()

        feature= {
            'full_img': _bytes_feature(full_img)
        }

        # Create a Features message using tf.train.Example.
        example_proto= tf.train.Example(features= tf.train.Features(feature= feature))
        return example_proto.SerializeToString()

    def tf_serialize_example(l,f):

        tf_string= tf.py_function(
            serialize_exemple,
            (l,f), # pass these args to the above function
            tf.string) # the return type is tf.string
        return tf.reshape(tf_string, ())

    serialized_dataset= dataset.map(tf_serialize_example)

    filename= 'trainData.tfrecord'
    writer= tf.data.experimental.TFRecordWriter(filename)
    writer.write(serialized_dataset)

    print('finished')
    return None

# ----------------------------------------------------------------------------

def decode(filename):
    dataset= tf.data.TFRecordDataset(filename)
    feature= {'full_img': tf.io.FixedLenFeature([], tf.string)}

    def _parse_example(input):
        feature_dic= tf.io.parse_single_example(input,feature)
        feature_dic['full_img']= tf.reshape(tf.io.decode_raw(feature_dic['full_img'], tf.int16),[512,512])
        return feature_dic['full_img']

    dataset= dataset.map(_parse_example())

    def preprocess(full):
        full= tf.cast(full, dtype=tf.float32)
        f_min= tf.reduce_min(full)
        f_max= tf.reduce_max(full)
        full= (full - f_min) / (f_max - f_min)
        return full

    dataset= dataset.map(preprocess)
    return dataset


print(encode2TfRecord())
