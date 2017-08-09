# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 17:13:22 2017

@author: nivradmin
"""

""" 
Implementation of DDPG - Deep Deterministic Policy Gradient
Algorithm and hyperparameter details can be found here: 
    http://arxiv.org/pdf/1509.02971v2.pdf
The algorithm is tested on the Pendulum-v0 OpenAI gym task 
and developed with tflearn + Tensorflow
Author: Patrick Emami
"""
import tensorflow as tf
import numpy as np
import gym
from gym import wrappers
from collections import deque
import random

from ddpg import DDPG_model

# ==========================
#   Training Parameters
# ==========================

# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.0001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.001
# Discount factor
GAMMA = 0.99

# ===========================
#   Utility Parameters
# ===========================
# Render gym env during training
RENDER_ENV = True
# Use Gym Monitor
GYM_MONITOR_EN = False
# Gym environment
ENV_NAME = 'Pendulum-v0'
# Directory for storing gym results
MONITOR_DIR = './results/gym_ddpg'
# Directory for storing tensorboard summary results
SUMMARY_DIR = './results/tf_ddpg'
RANDOM_SEED = 1234
# Size of replay buffer
BUFFER_SIZE = 10000
MINIBATCH_SIZE = 64

# ===========================
#   Actor and Critic DNNs
# ===========================

from collections import deque
import random
import numpy as np

class ReplayBuffer(object):

    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences 
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, batch):
        if self.count < self.buffer_size: 
            self.buffer.append(batch)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(batch)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        s2_batch = np.array([_[3] for _ in batch])
        t_batch = np.array([_[4] for _ in batch])
        
        print(s_batch)
        return s_batch, a_batch, r_batch, s2_batch, t_batch

    def clear(self):
        self.deque.clear()
        self.count = 0

        
        
        


def train(env, model):

    # Initialize replay memory
    replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)

    for i in range(50000):

        s = env.reset()

        ep_reward = 0
        ep_ave_max_q = 0

        for j in range(1000): #max.ep-step

            if RENDER_ENV:
                env.render()
            
            
            s = [(None, s)] #a state for the memory is always (conv, ff), so as there is no conv, its (None, ss) in this case.
            a = model.inference(s) + (1. / (1. + i)) # Added exploration noise

            s2, r, t, info = env.step(a[0])

            
            
            replay_buffer.add((s, a, r, s2, t))

            
            if replay_buffer.size() > MINIBATCH_SIZE:
                batch = replay_buffer.sample_batch(MINIBATCH_SIZE)
                
                ep_ave_max_q += model.train_step(batch) 
                
            s = s2
            ep_reward += r

            if t:
                print('| Reward: %.2i' % int(ep_reward), " | Episode", i, '| Qmax: %.4f' % (ep_ave_max_q / float(j)))
                break

            
class dummy():
    def __init__(self):
        pass

    
    
def main(_):
    tf.reset_default_graph()

    env = gym.make(ENV_NAME)
    np.random.seed(RANDOM_SEED)
    tf.set_random_seed(RANDOM_SEED)
    env.seed(RANDOM_SEED)
    
    conf = dummy()
    conf.num_actions = env.action_space.shape[0]
    conf.action_bounds = [(-env.action_space.high, env.action_space.high)]
    conf.image_dims = (0,0)
    conf.target_update_tau = 0.001               
    conf.actor_lr = 0.0001
    conf.critic_lr = 0.001 
     
    myAgent = dummy()
    myAgent.ff_inputsize = env.observation_space.shape[0]
    myAgent.usesConv = False        
    myAgent.ff_stacked = False
    myAgent.ff_inputsize = 3

    model = DDPG_model(conf, myAgent, tf.Session())


    if GYM_MONITOR_EN:
        if not RENDER_ENV:
            env = wrappers.Monitor(
                env, MONITOR_DIR, video_callable=False, force=True)
        else:
            env = wrappers.Monitor(env, MONITOR_DIR, force=True)

    train(env, model)

    if GYM_MONITOR_EN:
        env.monitor.close()

        
        
if __name__ == '__main__':
    tf.app.run()