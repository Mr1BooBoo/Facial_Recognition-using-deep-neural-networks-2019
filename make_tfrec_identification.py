# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 12:00:23 2019

@author: bilal & Lect Univ Alexandra
"""

import os
import cv2
import tensorflow as tf
import numpy as np

#GENERAREA PERECHI GENUIN
all_imgs = []

for i in range(1,16):
    director = 'D:/Licenta/code/dataset/MSC1M/' + '0' * (4-len(str(i))) + str(i)
    directories = ['D:/Licenta/code/dataset/MSC1M/' + '0' * (4-len(str(i))) + str(i) for i in range(1, 16)]
    img_names = [os.path.join(director,n) for n in os.listdir(director) if 'bmp' in n]
    all_imgs.append(img_names)
    
    nr_total = len(img_names)
del director, img_names


imgs_label = [[all_imgs[i][j], i] for i in range(len(all_imgs)) for j in range(len(all_imgs[i]))]
del all_imgs

# copiate din siteul tensorflow 
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    """Returns a int64_list from a bool / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


filename = 'D:/Licenta/code/15_id.tfrecords'
writer = tf.python_io.TFRecordWriter(filename)


for i in range(len(imgs_label)):
    poza = cv2.imread(imgs_label[i][0])
    encoded = cv2.imencode(".jpeg", poza)[1].tostring()
    
    example = tf.train.Example(features=tf.train.Features(feature = {
            'image_raw': _bytes_feature(encoded),
            'label': _int64_feature(int(imgs_label[i][1]))
            }))
    writer.write(example.SerializeToString())



"""for x in range (len(all_imgs)):
    for j in range(len(all_imgs[x])):
        poza = cv2.imread(all_imgs[x][j])
        encoded_a = cv2.imencode(".jpeg",poza)[1].tostring()
    
        example = tf.train.Example(features=tf.train.Features(feature = {
            'image_raw': _bytes_feature(encoded_a),
            'label':_int64_feature(int(imgs_label[x][1]))
            }))
        writer.write(example.SerializeToString())"""


nr_imgs = sum(1 for _ in tf.python_io.tf_record_iterator(filename))



# citirea fisierului tfrecords
def parse(example_proto):
    
      features = {"image_raw": tf.FixedLenFeature((), tf.string, default_value=""),
                  "label": tf.FixedLenFeature((), tf.int64, default_value=0)}
      
      features = tf.parse_single_example(example_proto, features)
      img = tf.image.decode_jpeg(features['image_raw'])
      label = tf.cast(features['label'], tf.int64)
      
      return img, label

dataset = tf.data.TFRecordDataset('D:/Licenta/code/15_id.tfrecords')
dataset = dataset.map(parse)

iterator = dataset.make_one_shot_iterator()
#image_batch, label_batch = iterator.get_next()
image_1, label_batch = iterator.get_next()


with tf.Session() as sess:
    for i in range(nr_imgs):
        im1, l = sess.run([image_1, label_batch])

        if i%50==0:
            cv2.imshow('im1', im1)
            #cv2.imshow('im1', im2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print(l)