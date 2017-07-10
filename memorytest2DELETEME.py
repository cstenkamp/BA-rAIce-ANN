# -*- coding: utf-8 -*-

"""
Created on Sun Jul  9 22:42:59 2017

@author: csten_000
"""

from precisememory import Memory as Precisememory
from efficientmemory import Memory as Efficientmemory
import numpy as np

class RLConf:
    def __init__(self):
        self.savememorypath = "./"
        

class Containers():
    def __init__(self):
        self.keep_memory = True
        self.rl_conf = RLConf()
    

containers = Containers()
        

MEMSIZE = 350

m1 =   Precisememory(MEMSIZE, containers)

m2 = Efficientmemory(MEMSIZE, containers, 4) 


print(m2._pointer) #immer der in pointer bis pointer+3 ist falsch
for i in range(MEMSIZE):
    if m2[i] != False and m2[i] is not None:
        print(i, np.all(m1[i][0][0][0] == m2[i][0][0][0]))
    elif m2[i] == False:
        print(i, "corrupted")
    else:
        print(i, "empty")
#
#print("")
#print("")
#print("")
#
#print(m2[1])
#
#samples, batch = m2.sample(5)
#
#batch2 = m1.sampletest(samples)
#
#
#print(samples)
#print(batch == batch2)
