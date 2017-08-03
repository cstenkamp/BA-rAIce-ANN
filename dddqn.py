# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 15:38:25 2017

@author: nivradmin
"""

from __future__ import division


import random
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt 
import time

import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #so that TF doesn't show its warnings
import math
#====own classes====
from myprint import myprint as print
from utils import convolutional_layer, fc_layer, variable_summary




class DDDQN():
    
    def __init__(self, config, agent):  
        self.config = config
        self.agent = agent
        
        self.pretrain_iter    = 0
        self.pretrain_iter_tf = tf.Variable(tf.constant(self.pretrain_iter), dtype=tf.int32, name='pretrain_iter_tf', trainable=False)
        self.step    = 0
        self.step_tf = tf.Variable(tf.constant(self.step), dtype=tf.int32, name='step_tf', trainable=False)
        self.num_actions = self.config.steering_steps*4 if self.config.INCLUDE_ACCPLUSBREAK else self.config.steering_steps*3
        #TODO: die beiden gleich laden
        
        self.h_size = 512
        
        #THIS IS FORWARD STEP
        self.scalarInput =  tf.placeholder(shape=[None,10800],dtype=tf.float32)   #input der reinkommt ist jetzt 4-stacked 30*45 bilder
        self.imageIn = tf.reshape(self.scalarInput,shape=[-1,30,45,8])
        self.Qout, self.predict = self._inference(self.imageIn)
        
        #THIS IS SV_LEARN
        self.Qout_SM = tf.nn.softmax(self.Qout)
        self.targetA = tf.placeholder(shape=[None],dtype=tf.int32)
        self.targetA_OH = tf.one_hot(self.targetA, self.num_actions, dtype=tf.float32)
        self.sv_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.targetA_OH, logits=self.Qout))
        self.sv_OP = self._pre_training(self.sv_loss, self.config.pretrain_initial_lr) 
        
        
        #THIS IS DDDQN-LEARN
        #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
        #self.targetA = tf.placeholder(shape=[None],dtype=tf.int32)
        #self.targetA_OH = tf.one_hot(self.targetA, self.num_actions, dtype=tf.float32)
        self.compareQ = tf.reduce_sum(tf.multiply(self.Qout, self.targetA_OH), axis=1)
        self.td_error = tf.square(self.targetQ - self.compareQ)
        self.q_loss = tf.reduce_mean(self.td_error)
        self.q_trainer = tf.train.AdamOptimizer(learning_rate=0.00005)
        self.q_updateModel = self.q_trainer.minimize(self.q_loss)        
        
        
        
    def _inference(self, imageIn):
        self.conv1 = slim.conv2d(inputs=imageIn,num_outputs=32,kernel_size=[4,6],stride=[2,3],padding='VALID', biases_initializer=None, normalizer_fn=slim.batch_norm)
        self.conv2 = slim.conv2d(inputs=self.conv1,num_outputs=64,kernel_size=[4,4],stride=[2,2],padding='VALID', biases_initializer=None, normalizer_fn=slim.batch_norm)
        self.conv3 = slim.conv2d(inputs=self.conv2,num_outputs=64,kernel_size=[3,3],stride=[1,1],padding='VALID', biases_initializer=None, normalizer_fn=slim.batch_norm)
        self.conv4 = slim.conv2d(inputs=self.conv3,num_outputs=self.h_size,kernel_size=[4,4],stride=[1,1],padding='VALID', biases_initializer=None, normalizer_fn=slim.batch_norm)
        #original war (?, 20, 20, 32) - (?, 9, 9, 64) - (?, 7, 7, 64) - (?, 1, 1, 512) - (?, 256)
        #meins ist    (?, 14, 14, 32) - (?, 6, 6, 64) - (?, 4, 4, 64) - (?, 1, 1, 512) - (?, 256)
        
        #We take the output from the final convolutional layer and split it into separate advantage and value streams.
        self.streamAC,self.streamVC = tf.split(self.conv4,2,3) #two splits, dimension 3
        self.streamA = slim.flatten(self.streamAC)
        self.streamV = slim.flatten(self.streamVC)
        xavier_init = tf.contrib.layers.xavier_initializer()
        self.AW = tf.Variable(xavier_init([self.h_size//2,self.num_actions]))
        self.VW = tf.Variable(xavier_init([self.h_size//2,1]))
        self.Advantage = slim.batch_norm(tf.matmul(self.streamA,self.AW))
        self.Value = slim.batch_norm(tf.matmul(self.streamV,self.VW))
        
        #Then combine them together to get our final Q-values.
        Qout = self.Value + tf.subtract(self.Advantage,tf.reduce_mean(self.Advantage,axis=1,keep_dims=True))
        predict = tf.argmax(Qout,1)
        return Qout, predict
    

    def _pre_training(self, loss, init_lr):
       self.pretrain_learningrate = tf.Variable(tf.constant(self.config.pretrain_initial_lr), name='pretrain_learningrate', trainable=False)
       self.new_lr = tf.placeholder(tf.float32, shape=[])                      #diese und die nächste zeile nur nötig falls man per extra-aufruf die lr verändern will, 
       self.pt_lr_update = tf.assign(self.pretrain_learningrate, self.new_lr)  #so wie ich das mache braucht man die nicht.
       train_op = tf.train.AdamOptimizer(self.pretrain_learningrate).minimize(loss, global_step=self.pretrain_iter_tf)
       return train_op
        
        
    def sv_fill_feed_dict(self, inputs, targets, decay_lr = True): 
        feed_dict = {}
        if decay_lr:
            lr_decay = self.config.pretrain_lr_decay ** max(self.pretrain_iter-self.config.pretrain_lrdecayafter, 0.0)
            new_lr = max(self.config.pretrain_initial_lr*lr_decay, self.config.pretrain_minimal_lr)
            feed_dict[self.new_lr] = new_lr #TODO: eig müsste er else nicht die opt durchführen die zu updaten
        feed_dict[self.scalarInput] = inputs
        feed_dict[self.targetA] = targets
        return feed_dict       
        
        
        
        
        
def processState(states):
    return np.reshape(states,[10800])        
        
    
    

def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)
        
     
   
        
        
        
        
import config
sv_conf = config.Config()
rl_conf = config.RL_Config()
import read_supervised
from server import Containers; containers = Containers(); containers.sv_conf = sv_conf; containers.rl_conf = rl_conf
import dqn_rl_agent
myAgent = dqn_rl_agent.Agent(sv_conf, containers, rl_conf, True)
trackingpoints = read_supervised.TPList(sv_conf.LapFolderName, sv_conf.use_second_camera, sv_conf.msperframe, sv_conf.steering_steps, sv_conf.INCLUDE_ACCPLUSBREAK)
      
trackingpoints.reset_batch()
stateBatch, pastBatch = trackingpoints.next_batch(sv_conf, myAgent, 1)        
oldstates, _, actions, rewards, _ = read_supervised.create_QLearnInputs_from_SVStateBatch(stateBatch, pastBatch, myAgent)
   

def buffersample(batchsize):
    oldstates, actions, rewards, newstates, terminals = read_supervised.create_QLearnInputs_from_SVStateBatch(*trackingpoints.next_batch(sv_conf, myAgent, batchsize), myAgent)
    oldstates = [processState(i) for i in oldstates[0]]
    newstates = [processState(i) for i in newstates[0]]             
    return np.swapaxes(np.array([oldstates, actions, rewards, newstates, terminals]), 0, 1)








BATCHSIZE = 32   
y = .99
tau = 0.001 

tf.reset_default_graph()

mainQN = DDDQN(rl_conf, myAgent)
targetQN = DDDQN(rl_conf, myAgent)

saver = tf.train.Saver()

trainables = tf.trainable_variables()

targetOps = updateTargetGraph(trainables,tau)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        
        trackingpoints.reset_batch()
        trainBatch = buffersample(trackingpoints.numsamples)
        predict = sess.run(targetQN.predict,feed_dict={targetQN.scalarInput:np.vstack(trainBatch[:,0])})
        print("Iteration",i,"Accuracy",round(np.mean(np.array(trainBatch[:,1] == predict, dtype=int))*100, 2),"%")

        if i % 10 == 0:
            print(sess.run(targetQN.Qout,feed_dict={targetQN.scalarInput:np.vstack(trainBatch[:,0])}))
        
        targetQN.pretrain_iter += 1
        trackingpoints.reset_batch()
        while trackingpoints.has_next(BATCHSIZE):
            trainBatch = buffersample(BATCHSIZE)
    
#            feed_dict = targetQN.sv_fill_feed_dict(np.vstack(trainBatch[:,0]), trainBatch[:,1])
#            _, loss, _ = sess.run([targetQN.sv_OP, targetQN.sv_loss, targetQN.pt_lr_update], feed_dict=feed_dict)
            #print(loss)
        
        #print("Learning rate:",sess.run(targetQN.pretrain_learningrate))
         

            
            #Below we perform the Double-DQN update to the target Q-values
            Q1 = sess.run(mainQN.predict,feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,3])})
            Q2 = sess.run(targetQN.Qout,feed_dict={targetQN.scalarInput:np.vstack(trainBatch[:,3])})
            end_multiplier = -(trainBatch[:,4] - 1)
            doubleQ = Q2[range(BATCHSIZE),Q1]
            targetQ = trainBatch[:,2] + (y*doubleQ * end_multiplier)
            #Update the network with our target values.
            _ = sess.run(mainQN.q_updateModel, feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,0]),mainQN.targetQ:targetQ, mainQN.targetA:trainBatch[:,1]})
            
            updateTarget(targetOps,sess) #Update the target network toward the primary network.
    



exit()        
        
        
        
        
        
        
        
    