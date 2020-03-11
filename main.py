# -*- coding: utf-8 -*-

import tensorflow as tf
from data_set import data_set_dict
from data_prepare import get_data
from model import Net
import shutil

tf.logging.set_verbosity(tf.logging.INFO)
path = 'E:\\tsproject\\t_data'
data_name = 'Adiac'
data_set = data_set_dict[data_name]
retrain = False
learning_rate = 0.0001
batch_size = 4
steps = data_set.train_size*3

def main(_):
    train_set = [path + '\\' +  data_name + '\\' + data_name+'_TRAIN']
    test_set = [path  + '\\' +  data_name + '\\' + data_name+'_TEST']
    model_url = path + '\\'+'cnn' + '\\' + data_name+'\\'    
    best = 0.0   
    data,label =   get_data(train_set, data_set.length, data_set.classes_num, 
                         batch_size,False)
   
    if retrain:
        shutil.rmtree(model_url)
   
    model = Net()

    hps = {
        'learning_rate': learning_rate,
           }   

    estimator = tf.estimator.Estimator(model.model_fn, model_url, params=hps)
    logging_hook = tf.train.LoggingTensorHook({}, every_n_iter=100, at_end=True)

    for i in range(125):
        estimator.train(
        lambda: get_data(train_set, data_set.length, data_set.classes_num, 
                         batch_size,True),  
        [logging_hook],
        steps=steps)

        result= estimator.evaluate(
        lambda: get_data(test_set, data_set.length, data_set.classes_num, 
                        data_set.test_size, False), 
        steps=1)

        if best < result['accuracy']:
            best= result['accuracy']
        print('The best accuracy is',best)
        print('The best error is',1-best)               

if __name__ == '__main__':
     tf.app.run()
