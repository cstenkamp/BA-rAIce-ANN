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

from dddqn import DDDQN_model

# ==========================
#   Training Parameters
# ==========================


RENDER_ENV = True
ENV_NAME = 'MountainCar-v0'
BUFFER_SIZE = 10000
MINIBATCH_SIZE = 64


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
    # Initialize replay memory
    epsilon = 0.9
    lasta = None
    replay_buffer = ReplayBuffer(BUFFER_SIZE)
    for episode in range(5000):
        s = env.reset()
        s = (None, s) #a state for the memory is always (conv, ff), so as there is no conv, its (None, ss) in this case.
        ep_reward = 0
        ep_ave_max_q = 0
        for j in range(1000): #max.ep-step
            if RENDER_ENV:
                env.render()
            if np.random.random() < epsilon:
                a = env.action_space.sample()
            else:
                a = int(model.inference([s])[0]) 
                if a != lasta: 
                    print(a)
                lasta = a
            
            
            s2, r, t, info = env.step(a)
            s2 = (None, s2)
            replay_buffer.append((s, a, r, s2, t))
            if len(replay_buffer) > MINIBATCH_SIZE:
                batch = replay_buffer.sample(MINIBATCH_SIZE)
                ep_ave_max_q += model.q_train_step(batch) 
            s = s2
            ep_reward += r
            if t:
                print('| Reward: %.2i' % int(ep_reward), " | Episode", episode, '| Qmax: %.4f' % (ep_ave_max_q / float(j)))
                break
            
        if episode % 20 == 0:
            model.save()     
            
        epsilon -= 0.0025
        print("Epsilon:",epsilon)



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
    
    conf = dummy()
    conf.image_dims = (0,0)
    conf.target_update_tau = 0.001               
    conf.actor_lr = 0.0001
    conf.critic_lr = 0.001 
    conf.checkpoint_dir = ".\gym_chkpt2"
    conf.pretrain_sv_initial_lr = 0.0
    conf.initial_lr = 0.001
    conf.use_settozero = False
    conf.q_decay = 0.99
    
    myAgent = dummy()
    myAgent.ff_inputsize = env.observation_space.shape[0]
    myAgent.usesConv = False        
    myAgent.ff_stacked = False
    myAgent.ff_inputsize = len(env.observation_space.high)
    myAgent.folder = folder
    myAgent.conv_stacked = False
    myAgent.isSupervised = False

    model = DDDQN_model(conf, myAgent, tf.Session(), num_actions=3)
    model.initNet("noPreTrain")

    train(env, model)


        
        
if __name__ == '__main__':
    tf.app.run()