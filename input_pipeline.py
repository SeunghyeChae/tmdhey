import tensorflow as tf
import pydicom as dcm
import pathlib
import os
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#
# def encode2TfRecord():
#     flag= 1
#     image_path= '/media/qmia/tmdhey/LUNA/dcm_data/SS0/1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260'
#     if not os.path.exists(image_path):
#         print(image_path + 'not exist')
#         return
#     allFulldose= glob(os.path.join(image_path,"*.dcm"))
#
#     def path2img(imgpaths):
#         images= []
#         for imgPath in imgpaths:
#             image_bytes= dcm.dcmread(imgPath)
#
#             image= image_bytes.pixel_array
#             images.append(image)
#         return images
#
#     print(len(allFulldose))
#     fulldose_imgs = path2img(allFulldose)
#     fulldose_imgs = np.array(fulldose_imgs,dtype='int16')
#     dataset = tf.data.Dataset.from_tensor_slices(fulldose_imgs)
#
#
#     def _bytes_feature(value):
#         '''Returns a bytes_list from a string / byte.'''
#         if  isinstance(value, type(tf.constant(0))):
#             value= value.numpy() # ByteList won't unpack a string from an EagerTensor
#         return tf.train.Feature(bytes_list= tf.train.BytesList(value=[value]))
#
#
#     def serialize_exemple(full_img):
#         '''Creates a tf.Example message ready to be written to a file'''
#         # Creates a dictionary message readyto be written to a file
#         # data type
#
#         full_img= full_img.numpy().tobytes()
#
#         feature= {
#             'full_img': _bytes_feature(full_img)
#         }
#
#         # Create a Features message using tf.train.Example.
#         example_proto= tf.train.Example(features= tf.train.Features(feature= feature))
#         return example_proto.SerializeToString()
#
#
#
#
#     def tf_serialize_example(f):
#
#         tf_string= tf.py_function(
#             serialize_exemple,
#             f, # pass these args to the above function
#             tf.string) # the return type is tf.string
#         return tf.reshape(tf_string, ())
#
#     serialized_dataset= dataset.map(tf_serialize_example)
#
#     filename= 'trainData.tfrecord'
#     writer= tf.data.experimental.TFRecordWriter(filename)
#     writer.write(serialized_dataset)
#
#     print('finished')
#     return None
#
# # ----------------------------------------------------------------------------
#
# def decode(filename):
#     dataset= tf.data.TFRecordDataset(filename)
#     feature= {'full_img': tf.io.FixedLenFeature([], tf.string)}
#
#     def _parse_example(input):
#         feature_dic= tf.io.parse_single_example(input,feature)
#         feature_dic['full_img']= tf.reshape(tf.io.decode_raw(feature_dic['full_img'], tf.int16),[512,512])
#         return feature_dic['full_img']
#
#     dataset= dataset.map(_parse_example())
#
#     def preprocess(full):
#         full= tf.cast(full, dtype=tf.float32)
#         f_min= tf.reduce_min(full)
#         f_max= tf.reduce_max(full)
#         full= (full - f_min) / (f_max - f_min)
#         return full
#
#     dataset= dataset.map(preprocess)
#     return dataset
#
#
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# # Encode TFRecord
# import os
# from tqdm import tqdm
# from glob import glob
# import random
# import tensorflow as tf
#
# dataset_path= '/media/qmia/tmdhey/LUNA/dcm_data/SS0/1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260'
#
# def _bytes_feature(value):
#     """Returns a bytes_list from a string / byte."""
#     if isinstance(value, type(tf.constant(0))):
#         value = value.numpy()
#     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
#
#
# def _float_feature(value):
#     """Returns a float_list from a float / double."""
#     return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
#
#
# def _int64_feature(value):
#     """Returns an int64_list from a bool / enum / int / uint."""
#     return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
#
#
# def make_example(img_str, source_id, filename):
#     # Create a dictionary with features that may be relevant.
#     feature = {'image/source_id': _int64_feature(source_id),
#                'image/filename': _bytes_feature(filename),
#                'image/encoded': _bytes_feature(img_str),
#                # -----------------------------------------
#                'image/height': _int64_feature(shape[0]),
#                'image/width': _int64_feature(shape[1]),
#                'image/channels': _int64_feature(shape[2]),
#                'image/shape': _int64_feature(shape),
#                'image/image_data':_bytes_feature(image_data.tostring()),
#                'image/superpixels':_bytes_feature(superpixels.tostring()),
#                'image/mask_instance':_bytes_feature(mask_instance.tostring()),
#                'image/mask_class':_bytes_feature(mask_class.tostring()),
#                'image/class_labels':_int64_feature(class_labels),
#                'image/instance_labels':_int64_feature(instance_labels)}
#
#     return tf.train.Example(features=tf.train.Features(feature=feature))
#
#
# def main(dataset_path, output_path):
#     samples = []
#     print("Reading data list...")
#     for id_name in tqdm(os.listdir(dataset_path)):
#         img_paths = glob(os.path.join(dataset_path, id_name, '*.dcm'))
#         for img_path in img_paths:
#             filename = os.path.join(id_name, os.path.basename(img_path))
#             samples.append((img_path, id_name, filename))
#     random.shuffle(samples)
#
#     print("Writing tfrecord file...")
#     with tf.io.TFRecordWriter(output_path) as writer:
#         for img_path, id_name, filename in tqdm(samples):
#             tf_example = make_example(img_str=open(img_path, 'rb').read(),
#                                       source_id=int(id_name),
#                                       filename=str.encode(filename))
#             writer.write(tf_example.SerializeToString())
#
#
# if __name__ == "__main__":
#     main(dataset_path, "/media/qmia/tmdhey/LUNA/dcm_data//dataset_dcm_SS0_first.tfrecord")
#
#
# # ---------------------------------------------------------------------------
# # Decode
#
# tfr_path= "/media/qmia/tmdhey/LUNA/dcm_data//dataset_dcm_SS0_first.tfrecord"
#
# def decode(filename):
#     dataset= tf.data.TFRecordDataset(filename)
#     feature= {'full_img': tf.io.FixedLenFeature([], tf.string)}
#
#     def _parse_example(input):
#         feature_dic= tf.io.parse_single_example(input,feature)
#         feature_dic['full_img']= tf.reshape(tf.io.decode_raw(feature_dic['full_img'], tf.int16),[512,512])
#         return feature_dic['full_img']
#
#     dataset= dataset.map(_parse_example())
#
#     def preprocess(full):
#         full= tf.cast(full, dtype=tf.float32)
#         f_min= tf.reduce_min(full)
#         f_max= tf.reduce_max(full)
#         full= (full - f_min) / (f_max - f_min)
#         return full
#
#     dataset= dataset.map(preprocess)
#     return dataset
#
# print(decode(tfr_path))


import tensorflow as tf
import pydicom as dcm
import pathlib
import os
from glob import glob
import random
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import numpy as np

tfr_path = "/media/qmia/tmdhey/LUNA/dcm_data/"
dataset_path = '/media/qmia/tmdhey/LUNA/dcm_data/SS0/1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260'


def path2img(imgpaths):
    images = []
    for imgPath in imgpaths:
        image_bytes = dcm.dcmread(imgPath)

        image = image_bytes.pixel_array
        images.append(image)
    return images


def encode2TfRecord():
    flag = 1
    image_path = '/media/qmia/tmdhey/LUNA/dcm_data/SS0/1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260'

    if not os.path.exists(image_path):
        print(image_path + 'not exist')
        return

    allFulldose = glob(os.path.join(image_path, "*.dcm"))

    print(len(allFulldose))
    fulldose_imgs = path2img(allFulldose)
    fulldose_imgs = np.array(fulldose_imgs, dtype='float64')
    # dataset = tf.data.Dataset.from_tensor_slices((fulldose_imgs))

    # for elem in dataset:
    #   print(elem.numpy())

    def _float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(value):
        '''Returns a bytes_list from a string / byte.'''
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()  # ByteList won't unpack a string from an EagerTensor
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    # ------------------------------------------------------------------------------

    def serialize_exemple(full_img):
        '''Creates a tf.Example message ready to be written to a file'''
        # Creates a dictionary message readyto be written to a file
        # data type

        full_img = full_img.numpy().tobytes()

        feature = {
            'full_img': _bytes_feature(full_img),
            # --------------------------------------------------
            # 'image/source_id': tf.io.FixedLenFeature([full_img], tf.int64),
            # 'image/filename': tf.io.FixedLenFeature([full_img], tf.string),
            # 'image/encoded': tf.io.FixedLenFeature([full_img], tf.string),
            # # ---------------------------------------------------
            # 'image/height': _int64_feature(full_img.shape[0]),
            # 'image/width': _int64_feature(full_img.shape[1]),
            # 'image/channels': _int64_feature(full_img.shape[2]),
            # 'image/shape': _int64_feature(full_img.shape),
            # 'image/image_data': _bytes_feature(full_img.image_data.tostring()),
            # 'image/superpixels': _bytes_feature(full_img.superpixels.tostring()),
            # 'image/mask_instance': _bytes_feature(full_img.mask_instance.tostring()),
            # 'image/mask_class': _bytes_feature(full_img.mask_class.tostring()),
            # 'image/class_labels': _int64_feature(full_img.class_labels),
            # 'image/instance_labels': _int64_feature(full_img.instance_labels)
        }

        # Create a Features message using tf.train.Example.
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    def main(dataset_path, output_path):
        samples = []
        print("Reading data list...")
        for id_name in tqdm(os.listdir(dataset_path)):
            img_paths = glob(os.path.join(dataset_path, id_name, '*.dcm'))
            for img_path in img_paths:
                filename = os.path.join(id_name, os.path.basename(img_path))
                samples.append((img_path, id_name, filename))
        # random.shuffle(samples)

        print("Writing tfrecord file...")
        with tf.io.TFRecordWriter(output_path) as writer:
            for img_path, id_name, filename in tqdm(samples):
                tf_example = serialize_exemple(img_str=open(img_path, 'rb').read(),
                                               source_id=int(id_name),
                                               filename=str.encode(filename))
                writer.write(tf_example.SerializeToString())

    if __name__ == "__main__":
        main(dataset_path, tfr_path + "dataset_dcm_SS0_first.tfrecord")


encode2TfRecord()