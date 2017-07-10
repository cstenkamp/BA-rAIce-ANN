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
        
m1 =   Precisememory(5, containers)
m2 = Efficientmemory(5, containers, 4) 


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



oldstate6 = ([["vision31"],["vision41"],["vision51"],["vision61"]], "speed6")
action6 = "action6"
reward6 = "reward6"
newstate6 = ([["vision41"],["vision51"],["vision61"],["vision71"]], "speed7")



oldstate7 = ([["vision41"],["vision51"],["vision61"],["vision71"]], "speed7")
action7 = "action7"
reward7 = "reward7"
newstate7 = ([["vision51"],["vision61"],["vision71"],["vision81"]], "speed8")



oldstate8 = ([["vision51"],["vision61"],["vision71"],["vision81"]], "speed8")
action8 = "action8"
reward8 = "reward8"
newstate8 = ([["vision61"],["vision71"],["vision81"],["vision91"]], "speed9")



oldstate9 = ([["vision61"],["vision71"],["vision81"],["vision91"]], "speed9")
action9 = "action9"
reward9 = "reward9"
newstate9 = ([["vision71"],["vision81"],["vision91"],["vision101"]], "speed10")


m1.append([oldstate1, action1, reward1, newstate1, False]) 
m1.append([oldstate2, action2, reward2, newstate2, False]) 
m1.append([oldstate3, action3, reward3, newstate3, False]) 
m1.append([oldstate4, action4, reward4, newstate4, False]) 
m1.append([oldstate5, action5, reward5, newstate5, False]) 
m1.append([oldstate6, action6, reward6, newstate6, False]) 
m1.append([oldstate7, action7, reward7, newstate7, False]) 
m1.append([oldstate8, action8, reward8, newstate8, False]) 
m1.append([oldstate9, action9, reward9, newstate9, False]) 

m2.append([oldstate1, action1, reward1, newstate1, False]) 
m2.append([oldstate2, action2, reward2, newstate2, False]) 
m2.append([oldstate3, action3, reward3, newstate3, False]) 
m2.append([oldstate4, action4, reward4, newstate4, False]) 
m2.append([oldstate5, action5, reward5, newstate5, False]) 
m2.append([oldstate6, action6, reward6, newstate6, False]) 
m2.append([oldstate7, action7, reward7, newstate7, False]) 
m2.append([oldstate8, action8, reward8, newstate8, False]) 
m2.append([oldstate9, action9, reward9, newstate9, False]) 


print(m2._pointer) #immer der in pointer bis pointer+3 ist falsch
for i in range(m1.capacity):
    if m2[i] != False:
        print(i, m2[i] == m1[i])
    else:
        print(i, "corrupted")
