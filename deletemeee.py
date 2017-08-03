# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 18:00:19 2017

@author: nivradmin
"""


from gridworld import gameEnv

env = gameEnv(partial=False,size=5)

print(env.actions)