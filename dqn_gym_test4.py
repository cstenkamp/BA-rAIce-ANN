# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 17:13:22 2017

@author: nivradmin
"""
import sys; sys.path.append("models")
import tensorflow as tf
import numpy as np
import gym
from collections import deque
import random
import os
import tensorflow.contrib.slim as slim


# ==========================
#   Training Parameters
# ==========================


RENDER_ENV = False
ENV_NAME = 'FrozenLake-v0'
BUFFER_SIZE = 100000
MINIBATCH_SIZE = 64


update_freq = 4 #How often to perform a training step.
y = .99 #Discount factor on the target Q-values
startE = 0.1 #Starting chance of random action
endE = 0.001 #Final chance of random action
#annealing_steps = 200000 #How many steps of training to reduce startE to endE.
num_episodes = 4000 #How many episodes of game environment to train network with.
pre_train_steps = 100
max_epLength = 50 #The max allowed length of our episode.
load_model = False #Whether to load a saved model.
h_size = 32 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
tau = 0.001 #Rate to update target network toward primary network

def dense(x, units, activation=tf.identity, decay=None, minmax=None):
    if minmax is None:
        minmax = float(x.shape[1].value) ** -.5
    return tf.layers.dense(x, units,activation=activation, kernel_initializer=tf.random_uniform_initializer(-minmax, minmax), kernel_regularizer=decay and tf.contrib.layers.l2_regularizer(1e-2))



class ReplayBuffer(object):

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()

    def append(self, batch):
        if self.count < self.buffer_size: 
            self.buffer.append(batch)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(batch)

    def __len__(self):
        return self.count

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        s2_batch = np.array([_[3] for _ in batch])
        t_batch = np.array([_[4] for _ in batch])
        return s_batch, a_batch, r_batch, s2_batch, t_batch


        
class Qnetwork():
    def __init__(self):
        
        self.ff_inputs = tf.placeholder(shape=[None,16],dtype=tf.float32)
        self.W = tf.Variable(tf.random_uniform([16,4],0,0.01))
        self.Qout = tf.matmul(self.ff_inputs,self.W)
        self.predict = tf.argmax(self.Qout,1)
        
        #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.nextQ = tf.placeholder(shape=[None,4],dtype=tf.float32)
        loss = tf.reduce_sum(tf.square(self.nextQ - self.Qout))
        trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        self.updateModel = trainer.minimize(loss)

    

def train(env):
    tf.reset_default_graph()
    
    total_steps = 0
    
#    e = startE
#    stepDrop = (startE - endE)/annealing_steps
    e = 0.1
                
    replay_buffer = ReplayBuffer(BUFFER_SIZE)               
    lasthundredavg = deque(100*[0], 100)           
    net = Qnetwork()
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for episode in range(num_episodes):
            
            s = env.reset()
            ep_reward = 0
            ep_ave_max_q = 0
            
            for j in range(1000): #max.ep-step
            
                if RENDER_ENV:
                    env.render()
                          
                if np.random.rand(1) < e or total_steps < pre_train_steps:
                    a = [env.action_space.sample()]
                else:
                    a = sess.run(net.predict, feed_dict={net.ff_inputs:[np.identity(16)[s]]})
                
                s2, r, t, info = env.step(a[0])
                total_steps += 1
                replay_buffer.append((s, a, r, s2, t))
                
                if total_steps > pre_train_steps:
#                    if e > endE:
#                        e -= stepDrop
                    
                    if total_steps % (update_freq) == 0:
                        trainBatch = replay_buffer.sample(32) #Get a random batch of experiences.
                        b_s, b_a, b_r, b_s2, b_t = trainBatch      
#                        b_s = b_s[0]; b_a = b_a[0]; b_r = b_r[0]; b_s2 = b_s2[0]; b_t = b_t[0]
                                                 
                        allQ = sess.run(net.Qout,feed_dict={net.ff_inputs:[np.identity(16)[i] for i in b_s]})
                        
                        Q1 = sess.run(net.Qout,feed_dict={net.ff_inputs:[np.identity(16)[i] for i in b_s2]})
                        maxQ1 = np.max(Q1,axis=1)
                        
                        targetQ = allQ
                        end_multiplier = -(b_t-1)
                        
                        for i in range(len(b_t)):
                            targetQ[i,b_a[i]] = b_r[i] + y*maxQ1[i]*end_multiplier[i]
                            
                        
                        _,W1 = sess.run([net.updateModel,net.W],feed_dict={net.ff_inputs:np.identity(16)[b_s],net.nextQ:targetQ})

                s = s2
                ep_reward += r
                if t:
                    lasthundredavg.append(ep_reward)
                    avg = np.mean(lasthundredavg)
                    print('| Reward: %.2i' % int(ep_reward), " | Last100:",avg," | Episode", episode, '| Qmax: %.4f' % (ep_ave_max_q / float(j)),' Epsilon:',e)
                    e = 1./((episode/50) + 20)
                    break

            


    
    
def main(_):
    tf.reset_default_graph()

    env = gym.make(ENV_NAME)
    
    train(env)


        
        
if __name__ == '__main__':
    tf.app.run()