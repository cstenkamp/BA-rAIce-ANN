# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 22:37:40 2017

@author: csten_000
"""

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #so that TF doesn't show its warnings
import numpy as np
from tensorflow.contrib.framework import get_variables
from collections import namedtuple
#====own classes====
import read_supervised
from myprint import myprint as print
import config 
from utils import convolutional_layer, fc_layer, variable_summary


Network = namedtuple('Network', ['outputs', 'vars', 'ops', 'losses'])
                                      
class DDPG(object):
    
    ######methods for BUILDING the computation graph######
    def __init__(self, config, is_training=True):
        #builds the computation graph, using the next few functions (this is basically the interface)
        self.config = config
        self.iterations = 0
        self.stacksize = self.config.history_frame_nr*2 if self.config.use_second_camera else self.config.history_frame_nr
        
        self.prepareNumIters()
        
        
        
        

    #TODO: Batchnorm (alex) dropout (me), settozero (me), weight-decay only for online/target nets??
    #TODO: weight sharing between critic's convlayers & actors convlayers
    #soooooooooo, critic gets as input action+state, and returns a single (Q-) value
    def make_critic(self, inputs, actions, name='online', reuse=False, batchnorm=False, is_training=True):  
        self.trainvars = {}
        with tf.variable_scope(name, reuse=reuse) as scope:
            flat_size = math.ceil(self.config.image_dims[0]/4)*math.ceil(self.config.image_dims[1]/4)*64
            rs_input = tf.reshape(inputs, [-1, self.config.image_dims[0], self.config.image_dims[1], self.stacksize])
            #convolutional_layer(input_tensor, input_channels, kernel_size, stride, output_channels, name, act, is_trainable, batchnorm, is_training, weightdecay=False, pool=True, trainvars=None, varSum=None, initializer=None)
            conv1 = convolutional_layer(rs_input, self.stacksize, [5,5], 2, 32, "Conv1", tf.nn.relu, True, True, is_training, 0.01, False, self.trainvars, variable_summary, "fanin") #reduces to x//2*y//2
            conv2 = convolutional_layer(conv1, 32, [3,3], 1, 32, "Conv2", tf.nn.relu, True, True, is_training, 0.01, False, self.trainvars, variable_summary, "fanin")                #reduces to x//4*y//4
            conv2_flat =  tf.reshape(conv2, [-1, flat_size])
            #fc_layer(input_tensor, input_size, output_size, name, is_trainable, batchnorm, is_training, weightdecay=False, act=None, keep_prob=1, trainvars=None, varSum=None, initializer=None)
            fc1 = fc_layer(conv2_flat, flat_size, 200, "FC1", True, True, is_training, 0.01, tf.nn.relu, trainvars=self.trainvars, varSum=variable_summary, "fanin")
            fc1 = tf.concat([fc1, actions], 1) 
            fc2 = fc_layer(fc1, 200+len(actions), 200, "FC2", True, True, is_training, 0.01, tf.nn.relu, trainvars=self.trainvars, varSum=variable_summary, "fanin")
            q = fc_layer(fc2, 200, 1, "Final", True, True, is_training, 0.01, tf.nn.relu, trainvars=self.trainvars, varSum=variable_summary, tf.random_uniform_initializer(-0.0003, 0.0003))
            ops = scope.get_collection(tf.GraphKeys.UPDATE_OPS)
            losses = scope.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            return Network(tf.squeeze(q), get_variables(scope), ops, losses) 


   
  




    
    def set_placeholders(self, is_training, final_neuron_num):
        if self.config.history_frame_nr == 1:
            inputs = tf.placeholder(tf.float32, shape=[None, self.config.image_dims[0], self.config.image_dims[1]], name="inputs")  #first dim is none since inference has another batchsize than training
        else:
            inputs = tf.placeholder(tf.float32, shape=[None, self.stacksize, self.config.image_dims[0], self.config.image_dims[1]], name="inputs")  #first dim is none since inference has another batchsize than training
            
            
        return inputs, targets, speeds
    
    
#    def inference(self, inputs, spinputs, final_neuron_num, rl_not_trainables, for_training=False):



    
    def loss_func(self, logits, targets):

        
    
    def rl_loss_func(self, q, q_target):
    
    
    
    def training(self, loss, init_lr, optimizer_arg):

        
    def evaluation(self, argmaxs, targets):

    
    def prepareNumIters(self):
        self.numIterations = tf.Variable(tf.constant(0), trainable=False)
        self.newIters = tf.placeholder(tf.int32, shape=[]) 
        self.iterUpdate = tf.assign(self.numIterations, self.newIters)           
    
    ######methods for RUNNING the computation graph######
    
    def saveNumIters(self, session, value):
        session.run(self.iterUpdate, feed_dict={self.newIters: value})
        
    def restoreNumIters(self, session):
        return self.numIterations.eval(session=session)
        
    
    
    def train_fill_feed_dict(self, config, dataset, batchsize = 0, decay_lr = True):
        batchsize = config.batch_size if batchsize == 0 else batchsize
        _, visionvec, targets, speeds = dataset.next_batch(config, batchsize)
        if decay_lr:
            lr_decay = config.lr_decay ** max(self.iterations-config.lrdecayafter, 0.0)
            new_lr = max(config.initial_lr*lr_decay, config.minimal_lr)
        feed_dict = {self.inputs: visionvec, self.targets: targets, self.keep_prob: config.keep_prob, self.learning_rate: new_lr}
        if config.speed_neurons:
            feed_dict[self.speed_input] = speeds
        return feed_dict            


    def assign_lr(self, session, lr_value):
        session.run(self.lr_update, feed_dict={self.new_lr: lr_value})
     

    def run_train_epoch(self, session, dataset, summarywriter = None):

        return gesamtLoss
        
    
    def run_eval(self, session, dataset):            
        dataset.reset_batch()
        feed_dict = self.train_fill_feed_dict(self.config, dataset, dataset.numsamples) #would be terribly slow if we learned, but luckily we only evaluate. should be fine.
        accuracy, loss = session.run([self.accuracy, self.loss], feed_dict=feed_dict)
        return accuracy, loss, dataset.numsamples
            
            
    def run_inference(self, session, visionvec, otherinputs, hframes):
        if hframes > 1:
            if not type(visionvec[0]).__module__ == np.__name__:
                return False, (None, None) #dann ist das input-array leer
        else:
            if not type(visionvec).__module__ == np.__name__:
                return False, (None, None) #dann ist das input-array leer
            assert (np.array(visionvec.shape) == np.array(self.inputs.get_shape().as_list()[1:])).all()
            
        
        with tf.device("/cpu:0"):
            visionvec = np.expand_dims(visionvec, axis=0)
            feed_dict = {self.inputs: visionvec}  
            if self.config.speed_neurons:
                speed_disc = read_supervised.inflate_speed(otherinputs.SpeedSteer.velocity, self.config.speed_neurons, self.config.SPEED_AS_ONEHOT)
                feed_dict[self.speed_input] = np.expand_dims(speed_disc, axis=0)
            
            return True, session.run([self.argmax, self.q], feed_dict=feed_dict)

        
        
    def calculate_value(self, session, visionvec, speed, hframes):
        with tf.device("/cpu:0"):
            visionvec = np.expand_dims(visionvec, axis=0)
            feed_dict = {self.inputs: visionvec}  
            if self.config.speed_neurons:
               speed_disc = read_supervised.inflate_speed(speed, self.config.speed_neurons, self.config.SPEED_AS_ONEHOT)
               feed_dict[self.speed_input] = np.expand_dims(speed_disc, axis=0)
            
            return session.run(self.q_max, feed_dict=feed_dict)
            
        
       
def run_svtraining(config, dataset):
  
 



        
        
        
def main():
    conf = config.Config()
        
    trackingpoints = read_supervised.TPList(conf.LapFolderName, conf.use_second_camera, conf.msperframe, conf.steering_steps, conf.INCLUDE_ACCPLUSBREAK)
    print("Number of samples: %s | Tracking every %s ms with %s historyframes" % (trackingpoints.numsamples, str(conf.msperframe), str(conf.history_frame_nr)), level=6)
    run_svtraining(conf, trackingpoints)        
    
                
                
if __name__ == '__main__':    
    main()
    time.sleep(5)