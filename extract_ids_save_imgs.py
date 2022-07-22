# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 14:00:36 2019

@author: bilal
"""

import numpy as np
import tensorflow as tf
import cv2, os, sys


root_path = 'D:/Licenta/code/dataset/train.tfrecords'
#save_path = 'D:/Licenta/code/dataset/MSC1M/'

nr_examples = sum(1 for _ in tf.python_io.tf_record_iterator(root_path))
print(nr_examples, 'examples in tfrecord')
#nr_examples = 5822653

def parse(example_proto):
    
  features = {"image_raw": tf.FixedLenFeature((), tf.string, default_value=""),
              "label": tf.FixedLenFeature((), tf.int64, default_value=0)}
  
  features = tf.parse_single_example(example_proto, features)
  img = tf.image.decode_jpeg(features['image_raw'])
  label = tf.cast(features['label'], tf.int64)
  
  return img, label

dataset = tf.data.TFRecordDataset(root_path)
dataset = dataset.map(parse)

iterator = dataset.make_one_shot_iterator()
image_batch, label_batch = iterator.get_next()


d={}

    
# write all ids in d{}
with tf.Session() as sess:
    for i in range(nr_examples):
        l = sess.run(label_batch)
        if l not in d.keys():
            d[l] = [i]
        else:
            d[l].extend([i])
        
        if i%10000 == 0:
            print(i)
 

#write all ids with 385 or more imgs i d100{}  
#sunt 100 de ids cu 385 de poze :D


d1000={}
nr_id = 0
for k,v in d.items():
    if len(v)<148 and len(v)>124:
        d1000[k] = v
        print('added',k,v)
        nr_id += 1
        if nr_id == 1000:
            break

len(d1000.keys())

# save all imgs of the needed ids (slow method)
"""keys = [k for k in d100.keys()]

for i in range(1,101):
    director = '0' * (4-len(str(i))) + str(i)
    print(director)
    ids = i
    indices = d100[keys[i]]
    with tf.Session() as sess:
        for i in range(nr_examples):
            im, l = sess.run([image_batch, label_batch])
            if i in indices:
                #cv2.imshow('name', im[:,:,::-1])
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
                cv2.imwrite(save_path + director + '/' + str(i) + '.bmp', im[:,:,::-1])"""
   


#create 100 folders with names from 0001 to 0100 use once 
for i in range(1,1001):
    path = "D:/Licenta/code/dataset/MSC1000/" + "0" * (4-len(str(i))) + str(i)
    os.mkdir(path)
    del path





list_indices = []
list_classes = []

cl = 1
for key, value in d1000.items():
    list_classes += [cl]*len(value)
    list_indices += value
    cl += 1


with tf.Session() as sess:
    for i in range(nr_examples):
        im,l = sess.run([image_batch, label_batch])
        if i in list_indices:
            class_im = list_classes[list_indices.index(i)]
            path = "D:/Licenta/code/dataset/MSC1000/" + "0" * (4-len(str(class_im))) + str(class_im)
            cv2.imwrite(path + '/' + str(i) +'.bmp', im[:,:,::-1])
            print(class_im, i)
            #cv2.imwrite(save_path + director + '/' + str(i) + '.bmp', im[:,:,::-1])












