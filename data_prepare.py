# -*- coding: utf-8 -*-

import tensorflow as tf
import math

def get_data(filenames,data_len,class_num,batch_size,shuffle):
     length = data_len+1
     file_queue = tf.train.string_input_producer(filenames)
     reader = tf.TextLineReader()
     _, value = reader.read(file_queue)     
     record_defaults =  [[1.0]] * length
     data = tf.decode_csv(value, record_defaults)
     
     #label should minus one since one_hot function start from 0
     label = data[0:1]
     label = tf.stack(label)
     label= tf.cast(label, tf.int32)-1
     
     dataset = data[1:length]
     dataset = tf.stack(dataset)
     dataset = tf.transpose(dataset)
     dataset = tf.reshape(dataset, [(length-1),1])
     dataset= tf.cast(dataset, tf.float32)
     
     if shuffle:
             data_batch,label_batch = tf.train.shuffle_batch([dataset,label],
                                                        batch_size=batch_size,
                                                         capacity=10*batch_size,                                                       
                                                         min_after_dequeue=4*batch_size)                                                       
     else:
           
             data_batch,label_batch = tf.train.batch([dataset,label],
                                                batch_size=batch_size,                                                
                                                capacity=10*batch_size)
         
              
     label_batch = tf.one_hot(label_batch, depth=class_num) 
     label_batch = tf.cast(label_batch, dtype=tf.int32)
     label_batch = tf.reshape(label_batch, [batch_size, class_num])
     
     return  data_batch,label_batch
