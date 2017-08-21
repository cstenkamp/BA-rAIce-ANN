# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 18:15:28 2017

@author: csten_000
"""
import gym

# For practice purpose as part of Reinforcement Learning course.

## Optimized policy via Genetic Algorithm
best_policy = [0, 3, 3, 3, 0, 3, 0, 1, 3, 1, 0, 0, 2, 2, 1, 1] 
env = gym.make('FrozenLake-v0')
for i_episode in range(200):
    observation = env.reset()
    ep_reward = 0
    for t in range(100):
#        env.render()
        observation, reward, done, info = env.step(best_policy[observation])
        ep_reward += reward
        if done:
            print("Reward:",reward,"Episode finished after {} timesteps".format(t+1))
            break
env.close()



