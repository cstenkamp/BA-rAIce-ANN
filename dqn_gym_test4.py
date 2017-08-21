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
from dddqn import DDDQN_model


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
tau = 0.01 #Rate to update target network toward primary network


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


   
def train(env, model):
    tf.reset_default_graph()
    replay_buffer = ReplayBuffer(BUFFER_SIZE)
    
    total_steps = 0
    e = 0.1
    lasthundredavg = deque(100*[0], 100)           
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for episode in range(num_episodes):
            
            s = env.reset()
            s = (None, np.identity(model.agent.ff_inputsize)[s])
            ep_reward = 0
            ep_ave_max_q = 0
            
            for j in range(1000): #max.ep-step
                if RENDER_ENV:
                    env.render()
                
                if np.random.rand(1) < e or total_steps < pre_train_steps:
                    a = [env.action_space.sample()]
                else:
                    a = model.inference([s])[0]
                
                s2, r, t, info = env.step(a[0])
                s2 = (None, np.identity(model.agent.ff_inputsize)[s2])
                total_steps += 1
                replay_buffer.append((s, a[0], r, s2, t))
                
                if total_steps > pre_train_steps:
                    if total_steps % (update_freq) == 0:
                        trainBatch = replay_buffer.sample(MINIBATCH_SIZE) #Get a random batch of experiences.
                        model.q_train_step(trainBatch)

                s = s2
                ep_reward += r
                if t:
                    lasthundredavg.append(ep_reward)
                    avg = np.mean(lasthundredavg)
                    print('| Reward: %.2i' % int(ep_reward), " | Last100:",avg," | Episode", episode, '| Qmax: %.4f' % (ep_ave_max_q / float(j)),' Epsilon:',e)
                    e = 1./((episode/50) + 10)
                    break

            


    
def folder(x):
    if not os.path.exists(x):
        os.makedirs(x)
    return x

            
class dummy():
    def __init__(self):
        pass

    
    
def main(_):
    tf.reset_default_graph()

    env = gym.make(ENV_NAME)
    
    print(env.action_space)
    
    conf = dummy()
    conf.image_dims = (0,0)
    conf.target_update_tau = 0.01               
    conf.checkpoint_dir = ".\gym_chkpt3"
    conf.pretrain_sv_initial_lr = 0.0
    conf.initial_lr = 0.005
    conf.use_settozero = False
    conf.q_decay = 0.99
    
    
    myAgent = dummy()
    myAgent.usesConv = False        
    myAgent.ff_stacked = False
    myAgent.ff_inputsize = 16
    myAgent.folder = folder
    myAgent.conv_stacked = False
    myAgent.isSupervised = False

    model = DDDQN_model(conf, myAgent, tf.Session(), num_actions=4)
    model.initNet("noPreTrain")

    train(env, model)


        
        
if __name__ == '__main__':
    tf.app.run()