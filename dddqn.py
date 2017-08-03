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



from gridworld import gameEnv


env = gameEnv(partial=False,size=5)

class DDDQN():
    
    def __init__(self, config, agent):  
        NUM_ACTIONS = 21
        h_size = 512

        self.scalarInput =  tf.placeholder(shape=[None,10800],dtype=tf.float32)   #input der reinkommt ist jetzt 4-stacked 30*45 bilder
        self.imageIn = tf.reshape(self.scalarInput,shape=[-1,30,45,8])
        self.conv1 = slim.conv2d(inputs=self.imageIn,num_outputs=32,kernel_size=[4,6],stride=[2,3],padding='VALID', biases_initializer=None)
        self.conv2 = slim.conv2d(inputs=self.conv1,num_outputs=64,kernel_size=[4,4],stride=[2,2],padding='VALID', biases_initializer=None)
        self.conv3 = slim.conv2d(inputs=self.conv2,num_outputs=64,kernel_size=[3,3],stride=[1,1],padding='VALID', biases_initializer=None)
        self.conv4 = slim.conv2d(inputs=self.conv3,num_outputs=h_size,kernel_size=[4,4],stride=[1,1],padding='VALID', biases_initializer=None)
        #original war (?, 20, 20, 32) - (?, 9, 9, 64) - (?, 7, 7, 64) - (?, 1, 1, 512) - (?, 256)
        #meins ist    (?, 14, 14, 32) - (?, 6, 6, 64) - (?, 4, 4, 64) - (?, 1, 1, 512) - (?, 256)
        
        #We take the output from the final convolutional layer and split it into separate advantage and value streams.
        self.streamAC,self.streamVC = tf.split(self.conv4,2,3) #two splits, dimension 3
        self.streamA = slim.flatten(self.streamAC)
        self.streamV = slim.flatten(self.streamVC)
        xavier_init = tf.contrib.layers.xavier_initializer()
        self.AW = tf.Variable(xavier_init([h_size//2,NUM_ACTIONS]))
        self.VW = tf.Variable(xavier_init([h_size//2,1]))
        self.Advantage = tf.matmul(self.streamA,self.AW)
        self.Value = tf.matmul(self.streamV,self.VW)
        
        #Then combine them together to get our final Q-values.
        self.Qout = self.Value + tf.subtract(self.Advantage,tf.reduce_mean(self.Advantage,axis=1,keep_dims=True))
        self.predict = tf.argmax(self.Qout,1)
        
        #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, NUM_ACTIONS, dtype=tf.float32)
        
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)
        
        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)
        
        
        
        
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

mainQN = DDDQN(None, None)
targetQN = DDDQN(None, None)

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
        
        trackingpoints.reset_batch()
        while trackingpoints.has_next(BATCHSIZE):
            trainBatch = buffersample(BATCHSIZE)
    
            #Below we perform the Double-DQN update to the target Q-values
            Q1 = sess.run(mainQN.predict,feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,3])})
            Q2 = sess.run(targetQN.Qout,feed_dict={targetQN.scalarInput:np.vstack(trainBatch[:,3])})
            end_multiplier = -(trainBatch[:,4] - 1)
            doubleQ = Q2[range(BATCHSIZE),Q1]
            targetQ = trainBatch[:,2] + (y*doubleQ * end_multiplier)
            #Update the network with our target values.
            _ = sess.run(mainQN.updateModel, feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,0]),mainQN.targetQ:targetQ, mainQN.actions:trainBatch[:,1]})
            
            updateTarget(targetOps,sess) #Update the target network toward the primary network.
    



exit()        
        
        
        
        
        
        
        
        
#        
#        
#        
#        
#        
#        
#batch_size = 32 #How many experiences to use for each training step.
#update_freq = 4 #How often to perform a training step.
#y = .99 #Discount factor on the target Q-values
#startE = 1 #Starting chance of random action
#endE = 0.1 #Final chance of random action
#annealing_steps = 10000. #How many steps of training to reduce startE to endE.
#pre_train_steps = 10000 #How many steps of random actions before training begins.
#num_episodes = 1000 #How many EPISODES of game environment to train network with.
#max_epLength = 50 #The max allowed length of our episode.
#load_model = False #Whether to load a saved model.
#path = "./dqn" #The path to save our model to.
#h_size = 512 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
#tau = 0.001 #Rate to update target network toward primary network
#
#
#tf.reset_default_graph()
#mainQN = DDDQN(None, None)
#targetQN = DDDQN(None, None)
#
#
#saver = tf.train.Saver()
#
#trainables = tf.trainable_variables()
#
#targetOps = updateTargetGraph(trainables,tau)
#
#
##Set the rate of random action decrease. 
#e = startE
#stepDrop = (startE - endE)/annealing_steps
#
##create lists to contain total rewards and steps per episode
#stepsList = []
#rList = []
#total_steps = 0
#
##Make a path for our model to be saved in.
#if not os.path.exists(path):
#    os.makedirs(path)
#
#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#    alreadyrun = 0
#    if load_model == True:
#        ckpt = tf.train.get_checkpoint_state(path)
#        saver.restore(sess,ckpt.model_checkpoint_path)
#        alreadyrun = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1].split(".")[0])
#        print('Loading Model...', alreadyrun)
#    for episode in range(alreadyrun, num_episodes):
#        episodeBuffer = experience_buffer()
#        #Reset environment and get first new observation
#        s = env.reset()
#        s = processState(s)
#        d = False
#        rAll = 0
#        stepInEpi = 0
#        #The Q-Network
#        while stepInEpi < max_epLength: #If the agent takes longer than max_epLength moves to reach either of the blocks, end the trial.
#            stepInEpi+=1
#            #Choose an action by greedily (with e chance of random action) from the Q-network
#            if np.random.rand(1) < e or total_steps < pre_train_steps:
#                a = np.random.randint(0,4)
#            else:
#                a = sess.run(mainQN.predict,feed_dict={mainQN.scalarInput:[s]})[0]
#            s1,r,d = env.step(a)
#            s1 = processState(s1)
#            total_steps += 1
#            episodeBuffer.add(np.reshape(np.array([s,a,r,s1,d]),[1,5])) #Save the experience to our episode buffer.
#            
#            if total_steps > pre_train_steps:
#                if e > endE:
#                    e -= stepDrop
#                
#                if total_steps % (update_freq) == 0:
#                    trainBatch = myBuffer.sample(batch_size) #Get a random batch of experiences.
#                    #Below we perform the Double-DQN update to the target Q-values
#                    Q1 = sess.run(mainQN.predict,feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,3])})
#                    Q2 = sess.run(targetQN.Qout,feed_dict={targetQN.scalarInput:np.vstack(trainBatch[:,3])})
#                    end_multiplier = -(trainBatch[:,4] - 1)
#                    doubleQ = Q2[range(batch_size),Q1]
#                    targetQ = trainBatch[:,2] + (y*doubleQ * end_multiplier)
#                    #Update the network with our target values.
#                    _ = sess.run(mainQN.updateModel, \
#                        feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,0]),mainQN.targetQ:targetQ, mainQN.actions:trainBatch[:,1]})
#                    
#                    updateTarget(targetOps,sess) #Update the target network toward the primary network.
#            rAll += r
#            s = s1
#            
#            if d == True:
#
#                break
#        
#        myBuffer.add(episodeBuffer.buffer) #rather extend
#        stepsList.append(stepInEpi)
#        rList.append(rAll)
#        #Periodically save the model. 
#        if episode % 1000 == 0:
#            saver.save(sess,path+'/model-'+str(episode)+'.ckpt')
#            print("Saved Model")
#        if len(rList) % 10 == 0:
#            print(total_steps,np.mean(rList[-10:]), e)
#    saver.save(sess,path+'/model-'+str(episode)+'.ckpt')
#print("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")
#time.sleep(9999)
#
#
#rMat = np.resize(np.array(rList),[len(rList)//100,100])
#rMean = np.average(rMat,1)
#plt.plot(rMean)