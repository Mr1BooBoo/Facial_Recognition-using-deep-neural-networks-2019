# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 13:39:55 2019

@author: Alex
"""


import numpy as np
import tensorflow as tf
import time

#### configuration #################
db_filename = 'D:/Licenta/code/15_id.tfrecords'

batch_size = 64
epochs = 100

start= 8
embedding_size = 512
nr_classe = 100#85742

nr_ex = sum(1 for _ in tf.python_io.tf_record_iterator(db_filename))
print(nr_ex)
nr_examples = nr_ex #5822653#1218 # ms celeb


is_training=True

train_lr = 0.01
#num_validation = 1000 ####
num_validation = 5000










#################### data reading #######################
def parse(example_proto):
    
    features = {'image_raw': tf.FixedLenFeature([], tf.string), 'label': tf.FixedLenFeature([], tf.int64)}
    features = tf.parse_single_example(example_proto, features)
    img = tf.image.decode_jpeg(features['image_raw'])
    img = tf.cast(img, dtype=tf.float32)
    label = tf.cast(features['label'], tf.int64)
    
    return img, label

dataset = tf.data.TFRecordDataset(db_filename)
dataset = dataset.map(parse)
dataset = dataset.shuffle(buffer_size=1000)
validation = dataset.take(num_validation)
train = dataset.skip(num_validation)


train_dataset = dataset.batch(batch_size)
train_iterator = train_dataset.make_initializable_iterator()
train_next_sample = train_iterator.get_next()

validation_dataset = dataset.batch(batch_size)
validation_iterator = validation_dataset.make_initializable_iterator()
validation_next_sample = validation_iterator.get_next()










########################################### citirea tfrecord cu perechi #####################################

name = 'test.tfrecords'
path = 'D:/Licenta/code/'

nr_ex_pairs = sum(1 for _ in tf.python_io.tf_record_iterator(path+name))
print(nr_ex)

def parse_identification(example_proto):
    
      features = {"img_x": tf.FixedLenFeature((), tf.string, default_value=""),
                  "img_y": tf.FixedLenFeature((), tf.string, default_value=""),
                  "labels": tf.FixedLenFeature((), tf.int64, default_value=0)}
      
      features = tf.parse_single_example(example_proto, features)
      img1 = tf.image.decode_jpeg(features['img_x'])
      img2 = tf.image.decode_jpeg(features['img_y'])
      labels = tf.cast(features['labels'], tf.int64)
      
      return img1, img2, labels
  
dataset_test = tf.data.TFRecordDataset(path+name)
dataset_test = dataset_test.map(parse_identification)

test_dataset = dataset_test.batch(batch_size//2)
test_iterator = test_dataset.make_initializable_iterator()
test_next_sample = test_iterator.get_next()





# smallen learning rate

nr_filters = [start, start*2, start*4, start*8, start*16] # 32, 64, 128, 256, 512

with tf.device('/gpu:0'):
    initializer = tf.contrib.layers.xavier_initializer()
    
    x_placeholder = tf.placeholder(tf.float32, shape=[None, 112, 112,3])
    y_placeholder = tf.placeholder(tf.float32, shape=[None, nr_classe])
    
    ################################ first conv layer s2 ################################
    f1 = initializer([3,3,3,nr_filters[0]])
    b1 = initializer([nr_filters[0]])
    c1 = tf.nn.relu(tf.layers.batch_normalization((tf.nn.conv2d(input=x_placeholder, filter=f1, strides=[1,2,2,1], padding='SAME') + b1), training=is_training))
    # 56x56x 32
    print('layer shape',c1.get_shape())    
    
    
    # dw pw 32 s1
    ####################################### 1 ############################################
    shape = [3,3, nr_filters[0], 1]
    D1 = tf.Variable(initializer(shape))
    b2 = tf.Variable(initializer([nr_filters[0]])) 
    sc1 =  tf.nn.relu(tf.layers.batch_normalization((tf.nn.depthwise_conv2d(input=c1, filter=D1, strides=[1,1,1,1], padding = 'SAME') +b2), training=is_training))
    #pw
    shape = [1,1, nr_filters[0], nr_filters[1]]
    P1 = tf.Variable(initializer(shape))
    b3 = tf.Variable(initializer([nr_filters[1]])) 
    pw1 = tf.nn.relu(tf.layers.batch_normalization((tf.nn.conv2d(input = sc1, filter=P1, strides=[1,1,1,1], padding='SAME') +b3), training=is_training))
    # 56x56x64
    print('layer shape',pw1.get_shape())    
    
    
    # dw s2 pw 128
    ####################################### 2 ############################################
    shape = [3,3, nr_filters[1], 1]
    D2 = tf.Variable(initializer(shape))
    b4 = tf.Variable(initializer([nr_filters[1]])) 
    sc2 =  tf.nn.relu(tf.layers.batch_normalization((tf.nn.depthwise_conv2d(input=pw1, filter=D2, strides=[1,2,2,1], padding = 'SAME') +b4), training=is_training))
    #pw
    shape = [1,1, nr_filters[1], nr_filters[2]]
    P2 = tf.Variable(initializer(shape))
    b5 = tf.Variable(initializer([nr_filters[2]])) 
    pw2 = tf.nn.relu(tf.layers.batch_normalization((tf.nn.conv2d(input = sc2, filter=P2, strides=[1,1,1,1], padding='SAME') +b5), training=is_training))
    # 28x28x128
    #
    print('layer shape',pw2.get_shape())    
    
     
    # dw pw 128
    ####################################### 3 ############################################
    shape = [3,3, nr_filters[2], 1]
    D3 = tf.Variable(initializer(shape))
    b6 = tf.Variable(initializer([nr_filters[2]])) 
    sc3 =  tf.nn.relu(tf.layers.batch_normalization((tf.nn.depthwise_conv2d(input=pw2, filter=D3, strides=[1,1,1,1], padding = 'SAME') +b6), training=is_training))
    #pw
    print('layer shape',sc3.get_shape())    
    
    
    shape = [1,1, nr_filters[2], nr_filters[2]]
    P3 = tf.Variable(initializer(shape))
    b7 = tf.Variable(initializer([nr_filters[2]])) 
    pw3 = tf.nn.relu(tf.layers.batch_normalization((tf.nn.conv2d(input = sc3, filter=P3, strides=[1,1,1,1], padding='SAME') +b7), training=is_training))
    # 28x28x128
    print('layer shape',pw3.get_shape()) 
    
    
    
    # dw pw 256
    ####################################### 4 ############################################
    shape = [3,3, nr_filters[2], 1]
    D4 = tf.Variable(initializer(shape))
    b8 = tf.Variable(initializer([nr_filters[2]])) 
    sc4 =  tf.nn.relu(tf.layers.batch_normalization((tf.nn.depthwise_conv2d(input=pw3, filter=D4, strides=[1,2,2,1], padding = 'SAME') +b8), training=is_training))
    print('layer shape',sc4.get_shape()) #14x14x128
    
    #pw
    shape = [1,1, nr_filters[2], nr_filters[3]]
    P4 = tf.Variable(initializer(shape))
    b9 = tf.Variable(initializer([nr_filters[3]])) 
    pw4 = tf.nn.relu(tf.layers.batch_normalization((tf.nn.conv2d(input = sc4, filter=P4, strides=[1,1,1,1], padding='SAME') +b9), training=is_training))
    #14x14x256
    print('layer shape',pw4.get_shape()) 
    
    
    
    # dw pw 256
    ####################################### 5 ############################################
    shape = [3,3, nr_filters[3], 1]
    D5 = tf.Variable(initializer(shape))
    b10 = tf.Variable(initializer([nr_filters[3]])) 
    sc5 =  tf.nn.relu(tf.layers.batch_normalization((tf.nn.depthwise_conv2d(input=pw4, filter=D5, strides=[1,1,1,1], padding = 'SAME') +b10), training=is_training))
    
    #pw
    shape = [1,1, nr_filters[3], nr_filters[3]]
    P5 = tf.Variable(initializer(shape))
    b11 = tf.Variable(initializer([nr_filters[3]])) 
    pw5 = tf.nn.relu(tf.layers.batch_normalization((tf.nn.conv2d(input = sc5, filter=P5, strides=[1,1,1,1], padding='SAME') +b11), training=is_training))
    #14x14x256
    print('layer shape',pw5.get_shape()) 
    
    
    
    # dw pw 256
    ####################################### 6 ############################################
    shape = [3,3, nr_filters[3], 1]
    D6 = tf.Variable(initializer(shape))
    b12 = tf.Variable(initializer([nr_filters[3]])) 
    sc6 =  tf.nn.relu(tf.layers.batch_normalization((tf.nn.depthwise_conv2d(input=pw5, filter=D6, strides=[1,2,2,1], padding = 'SAME') +b12), training=is_training))
    #pw
    shape = [1,1, nr_filters[3], nr_filters[4]]
    P6 = tf.Variable(initializer(shape))
    b13 = tf.Variable(initializer([nr_filters[4]])) 
    pw6 = tf.nn.relu(tf.layers.batch_normalization((tf.nn.conv2d(input = sc6, filter=P6, strides=[1,1,1,1], padding='SAME') +b13), training=is_training))
    #7x7x256
    print('layer shape',pw6.get_shape())    
    
    
    # dw pw 256
    ####################################### x5 ############################################
    
    #### 1
    shape = [3,3, nr_filters[4], 1]
    D7 = tf.Variable(initializer(shape))
    b14 = tf.Variable(initializer([nr_filters[4]])) 
    sc7 =  tf.nn.relu(tf.layers.batch_normalization((tf.nn.depthwise_conv2d(input=pw6, filter=D7, strides=[1,1,1,1], padding = 'SAME') +b14), training=is_training))
    #pw
    shape = [1,1, nr_filters[4], nr_filters[4]]
    P7 = tf.Variable(initializer(shape))
    b15 = tf.Variable(initializer([nr_filters[4]])) 
    pw7 = tf.nn.relu(tf.layers.batch_normalization((tf.nn.conv2d(input = sc7, filter=P7, strides=[1,1,1,1], padding='SAME') +b15), training=is_training))
    
    
    
    #### 2
    shape = [3,3, nr_filters[4], 1]
    D8 = tf.Variable(initializer(shape))
    b16 = tf.Variable(initializer([nr_filters[4]])) 
    sc8 =  tf.nn.relu(tf.layers.batch_normalization((tf.nn.depthwise_conv2d(input=pw7, filter=D8, strides=[1,1,1,1], padding = 'SAME') +b16), training=is_training))
    #pw
    shape = [1,1, nr_filters[4], nr_filters[4]]
    P8 = tf.Variable(initializer(shape))
    b17 = tf.Variable(initializer([nr_filters[4]])) 
    pw8 = tf.nn.relu(tf.layers.batch_normalization((tf.nn.conv2d(input = sc8, filter=P8, strides=[1,1,1,1], padding='SAME') +b17), training=is_training))
    
    
    
    #### 3 
    shape = [3,3, nr_filters[4], 1]
    D9 = tf.Variable(initializer(shape))
    b18 = tf.Variable(initializer([nr_filters[4]])) 
    sc9 =  tf.nn.relu(tf.layers.batch_normalization((tf.nn.depthwise_conv2d(input=pw8, filter=D9, strides=[1,1,1,1], padding = 'SAME') +b18), training=is_training))
    #pw
    shape = [1,1, nr_filters[4], nr_filters[4]]
    P9 = tf.Variable(initializer(shape))
    b19 = tf.Variable(initializer([nr_filters[4]])) 
    pw9 = tf.nn.relu(tf.layers.batch_normalization((tf.nn.conv2d(input = sc9, filter=P9, strides=[1,1,1,1], padding='SAME') +b19), training=is_training))
    
    
    
    #### 4 
    shape = [3,3, nr_filters[4], 1]
    D10 = tf.Variable(initializer(shape))
    b20 = tf.Variable(initializer([nr_filters[4]])) 
    sc9 =  tf.nn.relu(tf.layers.batch_normalization((tf.nn.depthwise_conv2d(input=pw9, filter=D10, strides=[1,1,1,1], padding = 'SAME') +b20), training=is_training))
    #pw
    shape = [1,1, nr_filters[4], nr_filters[4]]
    P10 = tf.Variable(initializer(shape))
    b21 = tf.Variable(initializer([nr_filters[4]])) 
    pw10 = tf.nn.relu(tf.layers.batch_normalization((tf.nn.conv2d(input = sc9, filter=P10, strides=[1,1,1,1], padding='SAME') +b21), training=is_training))
    
    
    
    #### 5
    shape = [3,3, nr_filters[4], 1]
    D11 = tf.Variable(initializer(shape))
    b22 = tf.Variable(initializer([nr_filters[4]])) 
    sc11 =  tf.nn.relu(tf.layers.batch_normalization((tf.nn.depthwise_conv2d(input=pw10, filter=D11, strides=[1,1,1,1], padding = 'SAME') +b22), training=is_training))
    #pw
    shape = [1,1, nr_filters[4], nr_filters[4]]
    P11 = tf.Variable(initializer(shape))
    b23 = tf.Variable(initializer([nr_filters[4]])) 
    pw11 = tf.nn.relu(tf.layers.batch_normalization((tf.nn.conv2d(input = sc11, filter=P11, strides=[1,1,1,1], padding='SAME') +b23), training=is_training))
    
    print('layer shape',pw11.get_shape()) 
    #7x7x512


    shape = [3,3, nr_filters[4], 1]
    D12 = tf.Variable(initializer(shape))
    b24 = tf.Variable(initializer([nr_filters[4]])) 
    sc12 =  tf.nn.relu(tf.layers.batch_normalization((tf.nn.depthwise_conv2d(input=pw11, filter=D12, strides=[1,1,1,1], padding = 'SAME') +b24), training=is_training))
    #pw
    shape = [1,1, nr_filters[4], nr_filters[4]]
    P12 = tf.Variable(initializer(shape))
    b25 = tf.Variable(initializer([nr_filters[4]])) 
    pw12 = tf.nn.relu(tf.layers.batch_normalization((tf.nn.conv2d(input = sc12, filter=P12, strides=[1,1,1,1], padding='SAME') +b25), training=is_training))
    
    print('layer shape',pw11.get_shape()) 
    #7x7x512    
    
    
    # gdc
    shape = [7,7, nr_filters[4], 1]
    D13 = tf.Variable(initializer(shape))
    b26 = tf.Variable(initializer([nr_filters[4]])) 
    sc13 =  tf.nn.relu(tf.layers.batch_normalization((tf.nn.depthwise_conv2d(input=pw12, filter=D13, strides=[1,1,1,1], padding = 'VALID') +b26), training=is_training))
    print('layer shape',sc12.get_shape()) #1x1x512
    #pw
    shape = [1,1, nr_filters[4], embedding_size]
    P13 = tf.Variable(initializer(shape))
    b27 = tf.Variable(initializer([embedding_size])) 
    pw13 = tf.nn.conv2d(input = sc13, filter=P13, strides=[1,1,1,1], padding='SAME') +b27
    print('layer shape',pw12.get_shape()) 
    #1x1x512
        
    shape = pw13.get_shape()
    print('layer 9 (d+p) shape',shape)    
    
    # flatten
    f = tf.reshape(pw13, [-1, embedding_size])
    print('size f',f.get_shape())
    
    W = tf.Variable(initializer([embedding_size, nr_classe]))
    B = tf.Variable(initializer([ nr_classe]))
    print('size W',W.get_shape())
    
    
    z = tf.matmul(f, W) + B
    print('z shape', z.get_shape())
    
    a = tf.nn.softmax(z)
    
    y_pred = tf.argmax(a, axis=1)
    
    
    ## cost
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2( labels = y_placeholder, logits = z)
    cost = tf.reduce_mean(cross_entropy)
        
    # optimisation
    optimizer = tf.train.AdamOptimizer(train_lr, beta1=0.9, beta2=0.999, epsilon=0.1).minimize(cost)
        
    acc = tf.reduce_mean(tf.cast(tf.equal(y_pred, tf.argmax(y_placeholder, axis=1)), tf.float32))




sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

ant_acc = []
val_acc = []
epc_plt = []
    
nr_iterations_identification = nr_ex_pairs//(batch_size//2)
nr_iterations = nr_examples//batch_size
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    for e in range(epochs):
        is_training = True
        sess.run(train_iterator.initializer)
        sess.run(validation_iterator.initializer)

        start_time = time.time()
    
        #X_t, Y_t = np.zeros([test_final, 112,112,3]), np.zeros([test_final, nr_classe])
        cl = []
        
        for it in range(nr_iterations):
                
            X, Y = sess.run(train_next_sample)
            X = X/255. -0.5
            
            # to one hot and construct testing set
            y = np.zeros([batch_size, nr_classe])
            for b in range(len(Y)):
                y[b, Y[b]] = 1
            
            Y = y
            

            feed_dict_train = {x_placeholder:X,\
                               y_placeholder:Y}
            #print('x shape', X.shape)
            #z = sess.run(z, feed_dict = feed_dict_train)
            #print('z shape', z.shape)
            sess.run(optimizer, feed_dict=feed_dict_train)
            #print(it)
            
            if it%5000==0:
                X, Y = sess.run(validation_next_sample)
                X = X/255. -0.5
                
                # to one hot and construct testing set
                y = np.zeros([batch_size, nr_classe])
                for b in range(len(Y)):
                    y[b, Y[b]] = 1
                
                Y = y
                    
                feed_dict_train = {x_placeholder:X,\
                                   y_placeholder:Y}
                
                acv = sess.run(acc, feed_dict=feed_dict_train)
                print('validation accuracy', acv)
                val_acc.append(acv)
                
            if it%50==0:#! modifica to 10000
                print('start verification')
                sess.run(test_iterator.initializer)
                for it_id in range(nr_iterations_identification):
                    img1,img2,l = sess.run(test_next_sample)
                    img1 = img1/255. -0.5
                    img2 = img2/255. -0.5
                    print('img1 shape', img1.shape)
                    print('img2 shape', img2.shape)
                    
                    embedding_img1 = sess.run(pw13, feed_dict ={x_placeholder:img1} )
                    embedding_img2 = sess.run(pw13, feed_dict ={x_placeholder:img2} )
                    print('img1 embedding',embedding_img1.shape)
                    print('img2 embedding',embedding_img2.shape)
                    
                    
        acc_all_examples = 0
        sess.run(train_iterator.initializer)
        is_training = False
        for it in range(nr_iterations):
                
            X, Y = sess.run(train_next_sample)
            X = X/255.
            
            # to one hot and construct testing set
            y = np.zeros([batch_size, nr_classe])
            for b in range(len(Y)):
                y[b, Y[b]] = 1
                
            Y = y
            feed_dict_train = {x_placeholder:X,\
                               y_placeholder:Y}
            
            acb = sess.run(acc, feed_dict=feed_dict_train)
            acc_all_examples += acb

        print('epoch', e, 'acc', acc_all_examples/nr_iterations )
        print("--- minutes ---" ,(time.time() - start_time)/60)
        ant_acc.append(acc_all_examples/nr_iterations)
        epc_plt.append(e)



import matplotlib.pyplot as plt
plt.plot(epc_plt, ant_acc, 'b-')
plt.plot(epc_plt, val_acc, 'g-')
plt.savefig('D:/Licenta/code/plots/15ids_256bz_100eps_lightweight.png')
plt.show()








