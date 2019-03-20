# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 19:13:23 2019

@author: wgh
"""

import tensorflow as tf
import numpy as np

BOTTLENECK_TENSOR_SIZE = 2048
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
JPG_TENSOR_NAME='DecodeJpeg/contents:0'

#输入图像数据，得到softmax概率值（一个shape=(1,1008)的向量）
#predictions = sess.run(bottleneck_tensor,{'DecodeJpeg/contents:0': image_data})
def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def next_batch():
    
    filename_queue = tf.train.string_input_producer(["C://Users//wgh//Desktop//Oxford//jpg//Transferoxfordpicture.tfrecords"])# create a queue

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)#return file_name and file
    features = tf.parse_single_example(serialized_example,
                                   features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'image_raw' : tf.FixedLenFeature([], tf.string),
                                    })#return image and label
    
    img = tf.decode_raw(features['image_raw'], tf.uint8)
    img = tf.reshape(img, [28, 28, 3])  #reshape image to 500*500*3
    label = tf.cast(features['label'], tf.int32) #throw label tensor
    batch_size = 80
    capacity = 10000
    example_batch,label_batch = tf.train.shuffle_batch([img,label],batch_size = batch_size,capacity=capacity,min_after_dequeue=1280)
    return example_batch,label_batch

def getvalue(sess,filepath,img_list,NumOfImg):
    with tf.gfile.GFile(filepath, 'rb') as f:#'D://LearningModels//tensorflow_inception_graph.pb'
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
    bottleneck_tensor= sess.graph.get_tensor_by_name(BOTTLENECK_TENSOR_NAME) 
    listndarray=np.zeros((NumOfImg,2048))
    for i in range(NumOfImg):
        imag_tensor=tf.image.encode_jpeg(img_list[i])
        jpg_output_transfer = sess.run(bottleneck_tensor,{JPG_TENSOR_NAME: sess.run(imag_tensor)})
        listndarray[i]=jpg_output_transfer
    return listndarray

x_input=tf.placeholder(tf.float32,[80,2048])
y_previous=tf.placeholder(tf.int32,[80])
y_=tf.one_hot(y_previous,17)#one_hot稀疏编码
#bottleneck_tensor= tf.get_default_graph().get_tensor_by_name(BOTTLENECK_TENSOR_NAME)
#jpg_tensor= tf.get_default_graph().get_tensor_by_name(JPG_TENSOR_NAME)
W_fc=weight_variable([2048,17])
b_fc=bias_variable([17])
y_conv=tf.nn.softmax(tf.matmul(x_input,W_fc)+b_fc)
cross_entropy=-tf.reduce_sum(y_*tf.log(y_conv))
#cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv),reduction_indices=[1]))
#cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_conv, 1e-10, 1.0)))
trainstep = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
if __name__ == "__main__":
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.initialize_local_variables())
    data_batchs = next_batch()
    coord = tf.train.Coordinator()
    iterations=0
    threads = tf.train.start_queue_runners(sess,coord)
    
    try:
        while not coord.should_stop():
            data = sess.run(data_batchs)
            iterations+=1
            if(iterations==2):#iterations设置迭代的训练次数
                break
            #print(iterations,data[0],data[1])
            haha=getvalue(sess,'D://LearningModels//tensorflow_inception_graph.pb',data[0],80)
            #print(haha)
            for i in range(10):
                sess.run(trainstep,feed_dict={x_input:haha,y_previous:data[1]})                      
                print("step %d,trainning accuracy %g" %(i,sess.run(accuracy,feed_dict={x_input:haha,y_previous:data[1]})))
    except tf.errors.OutOfRangeError:
        print("complete")
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()