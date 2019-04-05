"""
@author:  wangquaxiu
@time:  2018/9/16 15:09

将图片转换为TFRecord格式

"""

import cv2
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import skimage.io as io
from skimage import transform

# 获取数据文件
def get_files(file_dir):
    """

    :param file_dir:
    :return: list of images and labels

    """
    cats = []
    label_cat = []
    dogs = []
    label_dog = []
    i = 0
    for file in os.listdir(file_dir):
        name = file.split('.')
        if name[0] == 'cat':
            cats.append(file_dir+'/'+file)
            if i == 0:
                print(file_dir+'/'+file)
                i = 1;
            label_cat.append(0)
        else:
            dogs.append(file_dir+'/'+file)
            label_dog.append(1)
    print("There are % d cats\nThere are %d dogs" %(len(cats), len(dogs)))

    print(cats[7])  # 最终得到的是图片名的列表

    # hstack函数按水平(按列顺序)把数组给堆叠起来
    image_list = np.hstack((cats, dogs))
    label_list = np.hstack((label_cat, label_dog))

    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]

    return image_list, label_list


"""生成整数型的属性"""
def int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

"""生成字符串型的属性"""
def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to_tfrecord(images, labels, save_dir, name):
    """convert all images and labels to ine tfrecord file
    Args:
        images: list of images directories , string type
        labels: list of labels, int type
        save_dir： the directory to save tfrecord file
        name: the name of tfrecord file, string type

    :return :
        no return
    Note:
        need time
    """
    fileName = (save_dir+name)
    n_samples = len(labels)

    if np.shape(images)[0] != n_samples:
        raise ValueError('Images size %d does not match label size %d.' %(images.shape[0], n_samples))

    # wait some time here, transforming need some time based on the size of your data.
    writer = tf.python_io.TFRecordWriter(fileName)
    print('\nTransform start.....')
    for i in np.arange(1, n_samples):
        try:
            image = cv2.imread(images[i])
            # cv2.imshow('test', image)
            # cv2.waitKey()
            if i % 100 == 0:
                print('transform '+ str(i) + 'steps')
            image = cv2.resize(image, (208, 208))
            b, g, r = cv2.split(image)
            rgb_image = cv2.merge([r, g, b])  # this is suitable
            image_raw = rgb_image.tostring()
            label = int(labels[i])
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': int64_feature(label),
                'image_raw': bytes_feature(image_raw)
            }))
            writer.write(example.SerializeToString())
        except IOError as e:
            print('Could not read:', images[i])
            print('error: %s' % e)
            print('Skip it!\n')
    writer.close()
    print('Transform  done.')


def read_and_decode(tfrecords_file, batch_size):
    '''read and decode tfrecord file, generate (image, label) batches
    Args:
        tfrecords_file: the directory of tfrecord file
        batch_size: number of images in each batch
    Returns:
        image: 4D tensor - [bat ch_size, width, height, channel]
        label: 1D tensor - [batch_size]
    '''
    # make an input queue from the tfrecord file
    filename_queue = tf.train.string_input_producer([tfrecords_file])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    img_features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
        }
    )
    image = tf.decode_raw(img_features['image_raw'], tf.uint8)

    image = tf.reshape(image, [208, 208, 3])
    label = tf.cast(img_features['label'], tf.float32)
    image = tf.image.per_image_standardization(image)
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=64,
                                              capacity=2000)
    return image_batch, tf.reshape(label_batch, [batch_size])
