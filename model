# -*- coding: utf-8 -*-

import tensorflow as tf

class Net:
    def __init__(self):
        self.__hps = None
        self.training = None
        self.__regularizer = None
        self.__initializer = None
        
        return

    def model_fn(self, features, labels, mode,  params):
        self.__hps = params
        print(params)
        self.training = True if mode == tf.estimator.ModeKeys.TRAIN else False
        logits = self.__net(features, labels.get_shape()[1])
       
        predictions = {
            'classes': tf.argmax(logits, 1),
            'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions)

        loss = tf.losses.softmax_cross_entropy(labels, logits)

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(
               self.__hps['learning_rate'])
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss, tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, train_op=train_op)

        return tf.estimator.EstimatorSpec(
            mode,
            predictions=predictions,
            loss=loss,
            eval_metric_ops={
                'accuracy':tf.metrics.accuracy( tf.argmax(labels, 1), predictions['classes'])
            })
            

    def __MACNN_block(self, x , kernels,reduce=16):
        cov1 = tf.layers.conv1d(x, kernels, 3,padding='same')

        cov2 = tf.layers.conv1d(x, kernels, 6,padding='same')

        cov3 = tf.layers.conv1d(x, kernels, 12, padding='same') 

        x = tf.concat([cov1,cov2,cov3], 2)
        x = tf.layers.batch_normalization(x, training=self.training)
        x = tf.nn.relu(x)
        y = tf.reduce_mean(x, 1)
        y = tf.layers.dense(y, units=int(kernels*3/reduce), use_bias=False, activation=tf.nn.relu)
        y = tf.layers.dense(y, units=kernels*3, use_bias=False, activation=tf.sigmoid)
        y = tf.reshape(y, [-1, 1,kernels*3])        
        return x*y
    def __MCNN_block(self, x , kernels, reduce=16):
        cov1 = tf.layers.conv1d(x, kernels, 3,padding='same')

        cov2 = tf.layers.conv1d(x, kernels, 6,padding='same')

        cov3 = tf.layers.conv1d(x, kernels, 12, padding='same') 

        x = tf.concat([cov1,cov2,cov3], 2)
        x = tf.layers.batch_normalization(x, training=self.training)
        x = tf.nn.relu(x)
          
        return x
    
    def __SACNN_block(self, x , kernels, reduce=16):
        x = tf.layers.conv1d(x, kernels, 3, padding='same')
        x = tf.layers.batch_normalization(x, training=self.training)
        x = tf.nn.relu(x)
        y = tf.reduce_mean(x, 1)
        y = tf.layers.dense(y, units=int(kernels/reduce), use_bias=False, activation=tf.nn.relu)
        y = tf.layers.dense(y, units=kernels, use_bias=False, activation=tf.sigmoid)
        y = tf.reshape(y, [-1, 1,kernels])        
        return x*y
    
    def __stack(self, x , loop_num, kernels, reduce):
        for i in range(loop_num):         
            x = self.__MACNN_block(x, kernels, reduce)
        return x

    def __net(self, x, classes_num):      
        x=self.__stack(x,2 , 64, 16)
        x=tf.layers.max_pooling1d(x,3,2,padding='same')
        x=self.__stack(x,2 , 128, 16)
        x=tf.layers.max_pooling1d(x,3,2,padding='same')
        x=self.__stack(x,2 , 256, 16)
            
        fc = tf.reduce_mean(x, 1)
        logits= tf.layers.dense(fc, units=classes_num)

        return logits
