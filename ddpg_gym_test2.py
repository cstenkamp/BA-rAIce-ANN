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

from ddpg import DDPG_model

# ==========================
#   Training Parameters
# ==========================


RENDER_ENV = True
ENV_NAME = 'CarRacing-v0'
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

        
        
        


def train(env, agent):
    # Initialize replay memory
    replay_buffer = ReplayBuffer(BUFFER_SIZE)
    total_steps = 0
    for i in range(50000):
        s = env.reset()
        s = (s/255, [])
        ep_reward = 0
        ep_ave_max_q = 0
        for j in range(1000): #max.ep-step
            total_steps += 1
            if RENDER_ENV:
                env.render()
            a = agent.inference([s])
#            a = [env.action_space.sample()]
            s2, r, t, info = env.step(a)
#            print("Action",[int(round(i*100))/100 for i in a],"Reward",round(r,2))
            s2 = (s2/255, [])
            replay_buffer.append((s, a, r, s2, t))
            if len(replay_buffer) > MINIBATCH_SIZE:
                if total_steps % 4 == 0:
                    batch = replay_buffer.sample(MINIBATCH_SIZE)
                    ep_ave_max_q += agent.train(batch) 
            s = s2
            ep_reward += r
            if t:
                print('| Reward: %.2i' % int(ep_reward), " | Episode", i, '| Qmax: %.4f' % (ep_ave_max_q / float(j)))
                break
        if i % 20 == 0:
            agent.model.save()        



def folder(x):
    if not os.path.exists(x):
        os.makedirs(x)
    return x

            
class dummy():
    def __init__(self):
        pass
    
    
class agent():
    def __init__(self, env, conf):
        self.conf = conf
        self.ff_inputsize = env.observation_space.shape[0]
        self.usesConv = True 
        self.conv_stacked = True       
        self.ff_stacked = False
        self.ff_inputsize = 0
        self.folder = folder
        self._noiseState = np.array([0]*self.conf.num_actions)
        self.model = DDPG_model(conf, self, tf.Session())
        self.model.initNet("noPreTrain")
        self.epsilon = 0.4
    
    def make_noisy(self, action):
        self._noiseState = self.conf.ornstein_theta * self._noiseState + np.random.normal(np.zeros_like(self._noiseState), self.conf.ornstein_std)
        action = action + self.epsilon * self._noiseState
        clip = lambda x,b: min(max(x,b[0]),b[1])
        action = [clip(action[i],self.conf.action_bounds[i]) for i in range(len(action))]
        return action
    
    def inference(self,s):
        return self.make_noisy(self.model.inference(s)[0][0])
    
    def train(self,batch):
        return self.model.q_train_step(batch)
    
    
    
    
def main(_):
    tf.reset_default_graph()

    env = gym.make(ENV_NAME)
    #bei CarRacing ist es steer, throttle, brake
    
    conf = dummy()
    conf.num_actions = env.action_space.shape[0]
    conf.action_bounds = list(zip(env.action_space.low, env.action_space.high))
    conf.image_dims = (96,96)
    conf.conv_stacksize= 3 #bilder sind RGB
    conf.target_update_tau = 0.001               
    conf.actor_lr = 0.0001
    conf.critic_lr = 0.001 
    conf.checkpoint_dir = ".\gym_chkpt4"
    conf.q_decay = 0.99 
    conf.ornstein_theta = 0.15
    conf.ornstein_std = 0.2
    
    myAgent = agent(env, conf)
    train(env, myAgent)


        
        
if __name__ == '__main__':
    tf.app.run()