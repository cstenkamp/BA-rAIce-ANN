# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 16:19:33 2017

@author: csten_000
"""
import numpy as np
from collections import deque
import random
import tensorflow as tf
import gym
import os

import sys; sys.path.append("models")
from dddqn import DDDQN_model
from ddpg import DDPG_model


ENV_NAME = "Pendulum-v0"
RENDER_ENV = True
IS_CONTINOUS = True
NUM_EPISODES = 50000
MAX_STEPS_EPISODE = 1000
SAVEALL = 20
SAVE_PATH = "data/"
MEMORYSIZE = 44000 #max for carracing!

class Memory():
    
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
    
    
###############################################################################    
    
  
class config():
    def __init__(self, env):
        if IS_CONTINOUS:
            self.num_actions = env.action_space.shape[0]
            self.action_bounds = list(zip(env.action_space.low, env.action_space.high))
            self.target_update_tau = 0.001 
            self.actor_lr = 0.0001
            self.critic_lr = 0.001 
        else:
            self.target_update_tau = 0.01
            if ENV_NAME == "FrozenLake-v0":
                self.dnum_actions = 4
            elif ENV_NAME == "MountainCar-v0":
                self.dnum_actions = 3
            
            
        #diese beiden sind nur für CarRacing
        self.image_dims = (96,96)
        self.conv_stacksize= 3 #bilder sind RGB
        
        self.checkpoint_dir = SAVE_PATH+"/gym_"+ENV_NAME+"_ckpt"
        self.q_decay = 0.99 
        self.ornstein_theta = 0.15
        self.ornstein_std = 0.2
        self.batchsize = 64
        self.use_settozero = False
        self.INCLUDE_ACCPLUSBREAK = True
        self.pretrain_sv_initial_lr = 0.0
        self.initial_lr = 0.005
        self.throttle_index = 1
        self.brake_index = 2
        self.steer_index = 0
        
        if ENV_NAME == "Pendulum-v0":
            self.update_frequency = 1   
        else:
            self.update_frequency = 4
       

        self.ff_inputsize = env.observation_space.shape[0]
        self.usesConv = False        
        self.ff_stacked = False
        self.ff_inputsize = len(env.observation_space.high)
    
    
    
    
    
class agent():
    def __init__(self, env, conf):
        self.conf = conf
        self.usesConv = True if ENV_NAME == "CarRacing-v0" else False
        self.conv_stacked = True     
        self.ff_stacked = False
        if ENV_NAME == "CarRacing-v0":
            self.ff_inputsize = 0
        elif ENV_NAME == "Pendulum-v0":
            self.ff_inputsize = len(env.observation_space.high)
        elif ENV_NAME == "MountainCar-v0":
            self.ff_inputsize = 2
        elif ENV_NAME == "FrozenLake-v0":
            self.ff_inputsize = 16
        else:
            self.ff_inputsize = len(env.observation_space.high)
            
        if IS_CONTINOUS:
            self._noiseState = np.array([0]*self.conf.num_actions)
            self.model = DDPG_model(conf, self, tf.Session())
            
        else:
            self.model = DDDQN_model(conf, self, tf.Session())
        self.model.initNet("noPreTrain")
         
        self.memory = Memory(MEMORYSIZE)
        self.isSupervised = False
        
      
            
    
    def make_noisy(self, action, epsilon):
        self._noiseState = self.conf.ornstein_theta * self._noiseState + np.random.normal(np.zeros_like(self._noiseState), self.conf.ornstein_std)
        action = action + 10*epsilon * self._noiseState
        clip = lambda x,b: min(max(x,b[0]),b[1])
        action = [clip(action[i],self.conf.action_bounds[i]) for i in range(len(action))]
        return action
    
    
    
    def inference(self,s,env,epsilon):
        if IS_CONTINOUS: 
            return self.make_noisy(self.model.inference([s])[0][0], epsilon)
        else:
            if np.random.rand(1) < epsilon:
                if ENV_NAME == "MountainCar-v0":
                    a = [2] if s[1][0] < -0.9 or s[1][1] > 0 or (abs(s[1][1]) < 0.001 and s[1][0] < -0.5) else [0]
                else:
                    a = [env.action_space.sample()]
            else:
                a = self.model.inference([s])[0]
        return a[0]

    
    def train(self,batch):
        return self.model.q_train_step(batch)
    
    def folder(self,x):
        if not os.path.exists(x):
            os.makedirs(x)
        return x
    
    
###############################################################################    
    
def preprocess(s,agent):
    if ENV_NAME == "CarRacing-v0":
        return (s/255, [])
    elif ENV_NAME == "FrozenLake-v0":
        return (None, np.identity(agent.ff_inputsize)[s])
    else:
        return (None, s)


    
    
def main(_):
    tf.reset_default_graph()
    env = gym.make(ENV_NAME)
    conf = config(env)
    myAgent = agent(env, conf)
    total_steps = 0
    lasthundredavg = deque(100*[0], 100)     
    epsilon = 1
    
    for episode in range(NUM_EPISODES):
        s = env.reset()
        s = preprocess(s,myAgent)
        ep_reward = 0
        ep_ave_max_q = 0
        for j in range(MAX_STEPS_EPISODE): #max.ep-step
            total_steps += 1
            if RENDER_ENV:
                env.render()
            a = myAgent.inference(s,env,epsilon)
#            a = [env.action_space.sample()]
            s2, r, t, info = env.step(a)
#            print("Action",[int(round(i*100))/100 for i in a],"Reward",round(r,2))
            s2 = preprocess(s2,myAgent)
            myAgent.memory.append((s, a, r, s2, t))
            if len(myAgent.memory) > myAgent.conf.batchsize:
                if total_steps % conf.update_frequency == 0:
                    batch = myAgent.memory.sample(myAgent.conf.batchsize)
                    ep_ave_max_q += myAgent.train(batch) 
            s = s2
            ep_reward += r
            if t:
                lasthundredavg.append(ep_reward)
                avg = np.mean(lasthundredavg)
                print('| Reward: %.2i' % int(ep_reward), " | Last100:",avg," | Episode", episode, '| Qmax: %.4f' % (ep_ave_max_q / float(j)),' Epsilon:',epsilon)
                if hasattr(myAgent, "__noiseState"):
                    myAgent._noiseState = np.zeros_like(myAgent._noiseState)
                if ENV_NAME == "MountainCar-v0":
                    epsilon = 1./((episode/100) + 1)
                else:
                    epsilon = 1./((episode/50) + 10)
                break
                
        if episode % SAVEALL == 0:
            myAgent.model.save()    


        
        
if __name__ == '__main__':
    tf.app.run()