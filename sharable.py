# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 15:01:58 2017

@author: nivradmin
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 13:41:09 2017

@author: csten_000
"""
import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #so that TF doesn't show its warnings
import math
#====own classes====
from myprint import myprint as print
from utils import convolutional_layer, fc_layer, variable_summary


###############################################################################
###############################################################################
                                      
class Model(object): #learning on gpu and application on cpu: https://stackoverflow.com/questions/44255362/tensorflow-simultaneous-prediction-on-gpu-and-cpu
    
    def __init__(self, config, agent):  
        self.config = config
        self.agent = agent
        final_neuron_num = self.config.steering_steps*4 if self.config.INCLUDE_ACCPLUSBREAK else self.config.steering_steps*3
        self.conv_stacksize = (self.config.history_frame_nr*2 if self.config.use_second_camera else self.config.history_frame_nr) if self.agent.conv_stacked else 1
        self.ff_stacksize = self.config.history_frame_nr if self.agent.ff_stacked else 1
        
        self.global_step = tf.Variable(0, dtype=tf.int32, name='global_step', trainable=False) #https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist.py 
            
        ####
        self.conv_inputs = tf.placeholder(tf.float32, shape=[None, self.conv_stacksize, self.config.image_dims[0], self.config.image_dims[1]], name="conv_inputs")  if self.agent.usesConv else None
        self.ff_inputs = tf.placeholder(tf.float32, shape=[None, self.ff_stacksize*self.agent.ff_inputsize], name="ff_inputs") if self.agent.ff_inputsize else None
        self.targets = tf.placeholder(tf.float32, shape=[None, final_neuron_num], name="sv_targets")    #TODO: DELETE THIS LINE, UND PACKE DAFÜR DIE VORHERIGE WIEDER HIER HIN!!!
        
        self.q, self.onehot, self.q_max, self.action = self._inference(self.conv_inputs, self.ff_inputs, final_neuron_num, True)   
        self.q_targets = tf.placeholder(tf.float32, shape=[None, final_neuron_num], name="q_targets")
        self.loss = tf.reduce_mean(tf.square(self.q_targets - self.q))
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.00001)
        self.train_op = self.trainer.minimize(self.loss)
        self.accuracy = self._evaluation(self.onehot, self.targets)
        
        
    
    
    def _inference(self, conv_inputs, ff_inputs, final_neuron_num, for_training=False): #stands_inputs existiert nur bei inference, sonst ists eh immer 0
        assert(conv_inputs is not None and ff_inputs is not None)
        self.trainvars = {}
        flat_size = 0

        ini = tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(self.config.image_dims[0]*self.config.image_dims[1])))
       
        rs_input = tf.reshape(conv_inputs, [-1, self.config.image_dims[0], self.config.image_dims[1], self.conv_stacksize]) #final dimension = number of color channels*number of stacked (history-)frames                  
        flat_size = math.ceil(self.config.image_dims[0]/4)*math.ceil(self.config.image_dims[1]/4)*64 #die /(2*2) ist wegen dem einen stride=2 
        #convolutional_layer(input_tensor, input_channels, kernel_size, stride, output_channels, name, act, is_trainable, batchnorm, is_training, weightdecay=False, pool=True, trainvars=None, varSum=None, initializer=None)
        conv1 = convolutional_layer(rs_input, self.conv_stacksize, [5,5], 1, 32, "Conv1", tf.nn.relu, True, False, for_training, False, True, self.trainvars, variable_summary, initializer=ini) #reduces to x//2*y//2
        conv2 = convolutional_layer(conv1, 32, [5,5], 1, 64, "Conv2", tf.nn.relu, True, False, for_training, False, True, self.trainvars, variable_summary, initializer=ini)                #reduces to x//4*y//4
        conv2_flat =  tf.reshape(conv2, [-1, flat_size])    #x//4*y//4+speed_neurons
        fc0 = tf.concat([conv2_flat, ff_inputs], 1) 
        flat_size += self.agent.ff_inputsize    
        
        #fc_layer(input_tensor, input_size, output_size, name, is_trainable, batchnorm, is_training, weightdecay=False, act=None, keep_prob=1, trainvars=None, varSum=None, initializer=None):
        fc1 = fc_layer(fc0, flat_size, final_neuron_num*20, "FC1", True, False, for_training, False, tf.nn.relu, 1, self.trainvars, variable_summary, initializer=ini)                 
        q = fc_layer(fc1, final_neuron_num*20, final_neuron_num, "FC2", True, False, for_training, False, None, 1, self.trainvars, variable_summary, initializer=ini) 

        y_conv = tf.nn.softmax(q)                                                          #[ 0.1,  0.2, ...]
        onehot = tf.one_hot(tf.argmax(y_conv, dimension=1), depth=final_neuron_num)        #[   0,    1, ...]
        q_max = tf.reduce_max(q, axis=1)                                                   #23.1
        action = tf.argmax(q, axis=1)                                                      #2
        
        return q, onehot, q_max, action
    
    
        
    def _evaluation(self, onehots, targets):
        #returns how many percent it got correct
        made = tf.cast(onehots, tf.bool)
        real = tf.cast(targets, tf.bool)
        compare = tf.reduce_mean(tf.cast(tf.reduce_all(tf.equal(made, real),axis=1), tf.float32))
        tf.summary.scalar('accuracy', compare)
        return compare
        
          

###############################################################################
    ######methods for RUNNING the computation graph######
###############################################################################
 
        
    
    def sv_fill_feed_dict(self, config, conv_inputs, other_inputs, targets, decay_lr = True, dropout = True): 
        feed_dict = {self.targets: targets}
        if self.agent.usesConv:
            feed_dict[self.conv_inputs] = conv_inputs
        if self.agent.ff_inputsize:
            feed_dict[self.ff_inputs] = other_inputs
        return feed_dict            



    def run_sv_eval(self, session, agent, stateBatch):    
        conv_inputs, other_inputs, _ = self.EnvStateBatch_to_AgentStateBatch(agent, stateBatch)
        targets = self.EnvStateBatch_to_AgentActionBatch(agent, stateBatch)
        feed_dict = self.sv_fill_feed_dict(self.config, conv_inputs, other_inputs, targets)        
        accuracy = session.run(self.accuracy, feed_dict=feed_dict)
        return accuracy, accuracy, accuracy
           
    
##################### RL-stuff ################################################
    
    
    def rl_fill_feeddict(self, conv_inputs, other_inputs):
        
        def is_inference(conv_inputs, other_inputs):
            return len(conv_inputs.shape) <= 3 if conv_inputs is not None else len(other_inputs.shape) <= 2
        
        feed_dict = {}
        if is_inference(conv_inputs, other_inputs):   
            conv_inputs = np.expand_dims(conv_inputs, axis=0) #expand_dims weil hier quasi batchsize=1
            other_inputs= np.expand_dims(other_inputs, axis=0) 
            
        if self.agent.usesConv:
            feed_dict[self.conv_inputs] = conv_inputs
        if self.agent.ff_inputsize:
            feed_dict[self.ff_inputs] = other_inputs 
        return feed_dict
        
        
    def run_inference(self, session, conv_inputs, other_inputs):        
        if conv_inputs is not None:
            assert type(conv_inputs[0]).__module__ == np.__name__
            assert (np.array(conv_inputs.shape) == np.array(self.conv_inputs.get_shape().as_list()[1:])).all()
        feed_dict = self.rl_fill_feeddict(conv_inputs, other_inputs)
        return session.run([self.onehot, self.q], feed_dict=feed_dict)

    
    def calculate_value(self, session, conv_inputs, other_inputs):
        feed_dict = self.rl_fill_feeddict(conv_inputs, other_inputs)
        return session.run(self.q_max, feed_dict=feed_dict)
    
        
    def rl_learn_forward(self, session, conv_inputs, other_inputs, following_conv_inputs, following_other_inputs):
        qs = session.run(self.q, feed_dict = self.rl_fill_feeddict(conv_inputs, other_inputs)) 
        max_qs = session.run(self.q_max, feed_dict = self.rl_fill_feeddict(following_conv_inputs, following_other_inputs))
        return qs, max_qs


    def rl_learn_step(self, session, conv_inputs, other_inputs, qs):
        feed_dict = self.rl_fill_feeddict(conv_inputs, other_inputs)
        feed_dict[self.q_targets] = qs
        session.run(self.train_op, feed_dict=feed_dict)    
    
    
############################## helper functions ###############################
    
    #takes as input a batch of ENV-STATES, and returns batch of AGENT-STATES
    def EnvStateBatch_to_AgentStateBatch(self, agent, stateBatch):
        presentStates = list(zip(*stateBatch))
        conv_inputs, other_inputs, _ = list(zip(*[agent.getAgentState(*presentState) for presentState in presentStates]))
        other_inputs = [agent.makeNetUsableOtherInputs(i) for i in other_inputs]
        return conv_inputs, other_inputs, False
        
    #takes as input batch of ENV-STATES, and return batch of AGENT-ACTIONS
    def EnvStateBatch_to_AgentActionBatch(self, agent, stateBatch):
        presentStates = list(zip(*stateBatch))
        targets = [agent.makeNetUsableAction(agent.getAction(*presentState)) for presentState in presentStates]
        return targets
       