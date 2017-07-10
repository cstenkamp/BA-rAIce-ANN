# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 22:42:59 2017

@author: csten_000
"""

from precisememory import Memory as Precisememory
from efficientmemory import Memory as Efficientmemory

class Containers():
    def __init__(self):
        self.keep_memory = False

containers = Containers()
        
m1 = Precisememory(4, containers)
m2 = Efficientmemory(4, containers, 4) 


oldstate1 = ([["vision11"],["vision12"],["vision13"],["vision14"]], "speed1")
action1 = "action1"
reward1 = "reward1"
newstate1 = ([["vision12"],["vision13"],["vision14"],["vision21"]], "speed2")




oldstate2 = ([["vision12"],["vision13"],["vision14"],["vision21"]], "speed2")
action2 = "action2"
reward2 = "reward2"
newstate2 = ([["vision13"],["vision14"],["vision21"],["vision31"]], "speed3")




oldstate3 = ([["vision13"],["vision14"],["vision21"],["vision31"]], "speed3")
action3 = "action3"
reward3 = "reward3"
newstate3 = ([["vision14"],["vision21"],["vision31"],["vision41"]], "speed4")




oldstate4 = ([["vision14"],["vision21"],["vision31"],["vision41"]], "speed4")
action4 = "action4"
reward4 = "reward4"
newstate4 = ([["vision21"],["vision31"],["vision41"],["vision51"]], "speed5")




oldstate5 = ([["vision21"],["vision31"],["vision41"],["vision51"]], "speed5")
action5 = "action5"
reward5 = "reward5"
newstate5 = ([["vision31"],["vision41"],["vision51"],["vision61"]], "speed6")




m1.append([oldstate1, action1, reward1, newstate1, False]) 
m1.append([oldstate2, action2, reward2, newstate2, False]) 
m1.append([oldstate3, action3, reward3, newstate3, False]) 
#m1.append([oldstate4, action4, reward4, newstate4, False]) 

m2.append([oldstate1, action1, reward1, newstate1, False]) 
m2.append([oldstate2, action2, reward2, newstate2, False]) 
m2.append([oldstate3, action3, reward3, newstate3, False]) 
m2.append([oldstate4, action4, reward4, newstate4, False]) 
#m2.append([oldstate5, action5, reward5, newstate5, False]) 

print(m2[0])

#print(m2[0])
#
#print(m1[0])
#
#for i in range(m1.capacity):
#    print(m2[i] == m1[i])
