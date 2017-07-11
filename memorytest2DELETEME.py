# -*- coding: utf-8 -*-

"""
Created on Sun Jul  9 22:42:59 2017

@author: csten_000
"""

from precisememory import Memory as Precisememory
from efficientmemory import Memory as Efficientmemory
import numpy as np
np.set_printoptions(threshold=np.nan)

class RLConf:
    def __init__(self):
        self.savememorypath = "./"
        self.image_dims = [30,45] 
        self.use_constantbutbigmemory = True
        self.visionvecdtype = np.int8

class Containers():
    def __init__(self):
        self.keep_memory = True
        self.rl_conf = RLConf()
    

containers = Containers()
        

MEMSIZE = 400

m1 =   Precisememory(MEMSIZE, containers)

#for i in range(2,116):
#    print(i, np.all(m1[i][0][0][0] == m1[i+1][0][0][1]), m1[i][4])  
#    #wir sehen, das gilt immer, bis auf beim Reset. Warum da nicht? weil er da den inputval RESETTET -> setzt auf nur nullen


for i in range(2,116):
    if m1[i-1][4]:
        print(i, np.all(m1[i][0][0][0] == m1[i][3][0][1]))

#print("")
#print("")   
#
#for i in range(2,116):
#    if m1[i-1][4]:
#        print(i, np.count_nonzero(m1[i][0][0][1]))
#        print(i, np.count_nonzero(m1[i][0][0][2]))
#        print(i, np.count_nonzero(m1[i][0][0][3]))
#        
##es gilt also: WENN er IM FRAME VORHER resettet hat, DANN steht in der history nur zeros
#
#
#




print("") 
m2 = Efficientmemory(MEMSIZE, containers, 4, containers.rl_conf.use_constantbutbigmemory) 

#for i in range(1,116):
#    if m1[i-1][4]:
#        print(i, np.all(m1[i][0][0][0] == m1[i][3][0][1]))
#        print(m1[i][0][0][0])
#        print(m1[i][3][0][1])
#        print(i, np.all(m2[i][0][0][0] == m2[i][3][0][1]))


#for i in range(114,120):
#    print(m1[i][1])
    
    

#for i in range(114,120):
#    print(i, "reset" if m1[i][4] else "")
#    print("s:      ", np.all(m1[i][0][0][0] == m2[i][0][0][0]), np.all(m1[i][0][0][1] == m2[i][0][0][1]), np.all(m1[i][0][0][2] == m2[i][0][0][2]), np.all(m1[i][0][0][3] == m2[i][0][0][3]), "|", m1[i][0][1] == m2[i][0][1])
#    print("s':", np.all(m1[i][3][0][0] == m2[i][3][0][0]), np.all(m1[i][3][0][1] == m2[i][3][0][1]), np.all(m1[i][3][0][2] == m2[i][3][0][2]), np.all(m1[i][3][0][3] == m2[i][3][0][3]), "|", m1[i][3][1] == m2[i][3][1])           


#print(m1[115][3][0][2])
#print(m2[115][3][0][1])
#print(np.all(m1[115][3][0][2] == m2[115][3][0][1]))

#for i in range(2,75):
#    if m1[i-1][4]:
##        print(m2[i][0])
##        print(m1[i][0])
#        print(i, np.all(m1[i][0][0][0] == m2[i][0][0][0]), m1[i][0][1] == m2[i][0][1])
#        print(np.all(m1[i][0][0][1] == m2[i][0][0][1]))
#        print(np.all(m1[i][0][0][2] == m2[i][0][0][2]))
#        print(np.all(m1[i][0][0][3] == m2[i][0][0][3]))
        
        
        
#Ok, I think I know what problem and solution are.
#Problem:
#Previously, I saved the entire visionvechistory of s and of s'.
#That is, I saved 1,2,3,4 & 2,3,4,5 for the first one and 2,3,4,5 & 3,4,5,6 for the second one.
#In the more efficient memory, I save 1,2,3,4 only the very first time, and from then on, only the one that I didnt save before: 5 for the first one and 6 for the second.
#This works perfectly for every situation... besides the ones after a inputval-reset: The very first value after inputval-reset is NOT saved which would be the 7 in our example.
#It wouldn't make sense to save it, because this is only part of s', and there is no s and a. Problem is however, that when saving the one right after, where there is a corresponding s, when I save the 8 I didn't save the 7 before...
#and python thus assumes the 6 was the last value before.
#Solution:
#overwrite the 6 with a 7 after a reset. One may now say that leads to problems when sampling that very value, but even though its will definitely be a different value than in the old memory, it will not matter...
#because we're talking about Q-learning here: the one before a reset is surely a terminal episode, that goes along with resets....
#and how does Q-learning work? the value of an action in s is r+y*maxval(s') BUT ONLY IF THE EPISODE IS NON-TERMINAL!!
#If the episode is terminal, the value is simply r! We don't look at the maxval of s' anyway, we DON'T LOOK AT S' ANYWAY IN THAT CASE --> OVERWRITING IT DOESN'T MATTER!!!!
        
        
#for i in range(2,116):
#    if m2[i-1][4]:
#        print(i, np.all(m1[i-1][3][0][0] == m2[i][0][0][0])) #das problem ist das inputval-reset, dann kriegt er eins zu wenig unter.
#        print(i, m1[i][0][0][0])
#        print(i, m2[i][0][0][0])

#
#for i in range(2,116):
#    if m2[i-1][4]:
#        print(i, np.count_nonzero(m2[i][0][0][0]))
#        print(i, np.count_nonzero(m2[i][0][0][1]))
#        print(i, np.count_nonzero(m2[i][0][0][2]))
#        print(i, np.count_nonzero(m2[i][0][0][3]))

allof = lambda t1, t2: np.all(t1[0] == t2[0]) and np.all(t1[1] == t2[1]) and np.all(t1[2] == t2[2]) and np.all(t1[3] == t2[3])
allequal = lambda m1, m2, i: allof(m1[i][0][0], m2[i][0][0]) and m1[i][0][1] == m2[i][0][1] and np.all(m1[i][1] == m2[i][1]) and m1[i][2] == m2[i][2] and allof(m1[i][3][0], m2[i][3][0]) and m1[i][3][1] == m2[i][3][1] and m1[i][4] == m2[i][4]

##
print("")
print("")
print(m2._pointer) #immer der in pointer bis pointer+3 ist falsch
for i in range(MEMSIZE):
    if m2[i] != False and m2[i] is not None:
        print(i, allequal(m1,m2, i), ("reset" if m1[(i)][4] else "") )
        if not allequal(m1,m2, i):
            print("s:      ", np.all(m1[i][0][0][0] == m2[i][0][0][0]), np.all(m1[i][0][0][1] == m2[i][0][0][1]), np.all(m1[i][0][0][2] == m2[i][0][0][2]), np.all(m1[i][0][0][3] == m2[i][0][0][3]), "|", m1[i][0][1] == m2[i][0][1])
            print("s':", np.all(m1[i][3][0][0] == m2[i][3][0][0]), np.all(m1[i][3][0][1] == m2[i][3][0][1]), np.all(m1[i][3][0][2] == m2[i][3][0][2]), np.all(m1[i][3][0][3] == m2[i][3][0][3]), "|", m1[i][3][1] == m2[i][3][1])           
    elif m2[i] == False:
        print(i, "corrupted")
    else:
        print(i, "empty")

print("")
print("")
print("")
##



###############################
#weiteres problem: wenn er freezed, vergisst er beim efficientmemory ein frame.

#for i in range(MEMSIZE):
#    if m2[i] != False and m2[i] is not None:
#        if not allequal(m1,m2, i) and not m1[i][4]:
#            print(i)
#            print("s:      ", np.all(m1[i][0][0][0] == m2[i][0][0][0]), np.all(m1[i][0][0][1] == m2[i][0][0][1]), np.all(m1[i][0][0][2] == m2[i][0][0][2]), np.all(m1[i][0][0][3] == m2[i][0][0][3]), "|", m1[i][0][1] == m2[i][0][1])
#            print("s':", np.all(m1[i][3][0][0] == m2[i][3][0][0]), np.all(m1[i][3][0][1] == m2[i][3][0][1]), np.all(m1[i][3][0][2] == m2[i][3][0][2]), np.all(m1[i][3][0][3] == m2[i][3][0][3]), "|", m1[i][3][1] == m2[i][3][1])           
#print("")
#print("")
#print("")
#
#print(np.all(m1[117][3][0][2] == m2[117][3][0][1]))





#print(m2[1])
#
#samples, batch = m2.sample(5)
#
#batch2 = m1.sampletest(samples)
#
#
#print(samples)
#print(batch == batch2)
