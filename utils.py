# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 22:15:47 2017

@author: csten_000
"""

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #so that TF doesn't show its warnings
import math
import numpy as np


def weight_variable(shape, name, weightdecay, initializer=None, is_trainable=True):
    if weightdecay > 0:    
        weight_decay = tf.constant(weightdecay, dtype=tf.float32) #https://stackoverflow.com/questions/36570904/how-to-define-weight-decay-for-individual-layers-in-tensorflow
        return tf.get_variable(name, shape, initializer=initializer, trainable=is_trainable, regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
    else:
        return tf.get_variable(name, shape, initializer=initializer, trainable=is_trainable)
    #only weight decay on weights, not biases! https://stats.stackexchange.com/questions/153605/no-regularisation-term-for-bias-unit-in-neural-network
		
def bias_variable(shape, name, is_trainable=True):
    #Since we're using ReLU neurons, it is also good practice to initialize them with a slightly positive initial bias to avoid "dead neurons"
    return tf.get_variable(name, shape, initializer=tf.constant_initializer(0.1), trainable=is_trainable)

def conv2d(x, filter, stride):
    return tf.nn.conv2d(x, filter, strides=[1, stride, stride, 1], padding='SAME') #https://www.tensorflow.org/api_docs/python/tf/nn/conv2d

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


#is_trainable = True if not self.is_reinforcement else not (name in rl_not_trainables)
def convolutional_layer(input_tensor, input_channels, kernel_size, stride, output_channels, name, act, is_trainable, batchnorm, is_training, weightdecay=False, pool=True, trainvars=None, varSum=None, initializer=None): #trainvars is call-by-reference-array, varSum is a function
    with tf.name_scope(name):
        if initializer == None:
            initializer = tf.truncated_normal_initializer(stddev=1.0)
        elif initializer == "fanin": #http://deeplearning.net/tutorial/lenet.html: fan-in is the number of inputs to a hidden unit
            initializer = tf.random_uniform_initializer(1/np.sqrt(float(input_channels*kernel_size[0]*kernel_size[1])), 1/np.sqrt(float(input_channels*kernel_size[0]*kernel_size[1])))
            
        if is_trainable:
            trainvars["W_%s" % name] = weight_variable([kernel_size[0], kernel_size[1], input_channels, output_channels], "W_%s" % name, weightdecay, initializer=initializer, is_trainable=True)
            varSum(trainvars["W_%s" % name])
            trainvars["b_%s" % name] = bias_variable([output_channels], "b_%s" % name, is_trainable=True)
            varSum(trainvars["b_%s" % name])
            h_act = act(conv2d(input_tensor, trainvars["W_%s" % name], stride) + trainvars["b_%s" % name])
        else:
            W = weight_variable([kernel_size[0], kernel_size[1], input_channels, output_channels], "W_%s" % name, weightdecay, initializer=initializer, is_trainable=False)
            b = bias_variable([output_channels], "b_%s" % name, is_trainable=False)
            h_act = act(conv2d(input_tensor, W, stride) + b)        
        if batchnorm:
            h_act = tf.layers.batch_normalization(h_act, training=is_training, epsilon=1e-7, momentum=.95)
        if pool:
            h_pool = max_pool_2x2(h_act)
        tf.summary.histogram("activations", h_pool)
        return h_pool
    

#is_trainable = True if not self.is_reinforcement else not (name in rl_not_trainables)
def fc_layer(input_tensor, input_size, output_size, name, is_trainable, batchnorm, is_training, weightdecay=False, act=None, keep_prob=1, trainvars=None, varSum=None, initializer=None):
    with tf.name_scope(name):
        if initializer == None:
            initializer = tf.truncated_normal_initializer(stddev=1.0)
        elif initializer == "fanin": #http://deeplearning.net/tutorial/lenet.html: fan-in is the number of inputs to a hidden unit
            initializer = tf.random_uniform_initializer(1/np.sqrt(float(input_size)), 1/np.sqrt(float(input_size)))
            
        if is_trainable:
            trainvars["W_%s" % name] = weight_variable([input_size, output_size], "W_%s" % name, weightdecay, is_trainable=True, initializer=initializer)
            varSum(trainvars["W_%s" % name])
            trainvars["b_%s" % name] = bias_variable([output_size], "b_%s" % name, is_trainable=True)
            varSum(trainvars["b_%s" % name])
            h_fc =  tf.matmul(input_tensor, trainvars["W_%s" % name]) + trainvars["b_%s" % name]
        else:
            W = weight_variable([input_size, output_size], "W_%s" % name, weightdecay, is_trainable=False, initializer=initializer)
            b = bias_variable([output_size], "b_%s" % name, is_trainable=False)
            h_fc =  tf.matmul(input_tensor, W) + b
        if act is not None:
            h_fc = act(h_fc)
        if is_trainable:
            tf.summary.histogram("activations", h_fc)
            if is_training:
                h_fc = tf.nn.dropout(h_fc, keep_prob) 
        if batchnorm:
            h_fc = tf.layers.batch_normalization(h_fc, training=is_training, epsilon=1e-7, momentum=.95) #training can be a python bool!
        return h_fc


def variable_summary(var, what=""):
    with tf.name_scope('summaries_'+what):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


###############################################################################
#alternatively to the direct TF-implementation of layers, one could have used the tf.layers as well:
#def dense(x, units, activation=None, decay=None, minmax=None):
#    if minmax is None:
#        minmax = float(x.shape[1].value) ** -.5
#    return tf.layers.dense(x, units, activation=activation, kernel_initializer=tf.random_uniform_initializer(-minmax, minmax),
#                           kernel_regularizer=decay and tf.contrib.layers.l2_regularizer(1e-3) )
#
#def conv(x, filters, kernelsize, padding="same", activation=tf.nn.reulu):
#    return tf.layers.conv2d(inputs=x, filters=filters, kernel_size=kernelsize, padding=padding, activation=activation)
#
#
#def pool(x, pool_size=[2,2], strides=2):
#     return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=strides)
    