# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 16:16:35 2017

@author: csten_000
"""


import os
import pickle
import threading
import shutil
import time
current_milli_time = lambda: int(round(time.time() * 1000))
import random
import numpy as np
#====own classes====
from myprint import myprint as print


#we want something with fast random access, so we take a list instead of a deque https://wiki.python.org/moin/TimeComplexity
#https://docs.python.org/3/library/collections.html#deque-objects


#TODO: not sure how thread-safe this is.. https://stackoverflow.com/questions/13610654/how-to-make-built-in-containers-sets-dicts-lists-thread-safe
class Memory(object):
    def __init__(self, capacity, containers, state_stacksize):
        self._lock = lock = threading.Lock()
        
        self.capacity = capacity
        self._state_stacksize = state_stacksize
        self._pointer = 0
        self._appendcount = 0
        self.containers = containers
        self.lastsavetime = current_milli_time()
        self._size = 0
        
        self._visionvecs = [None]*capacity
        self._speeds = [None]*capacity 
        self._actions = [None]*capacity
        self._rewards = [None]*capacity #np.zeros(capacity, dtype=np.float)
        self._fEnds = [None]*capacity #np.zeros(capacity, dtype=np.bool)
        #keine Folgestates, da die ja im n+1ten Element stecken
        
        if self.containers.keep_memory:
            corrupted = False
            if os.path.exists(self.containers.rl_conf.savememorypath+'memory.pkl'):
                try:
                    if os.path.getsize(self.containers.rl_conf.savememorypath+'memory.pkl') > 1024 and (os.path.getsize(self.containers.rl_conf.savememorypath+'memory.pkl') >= os.path.getsize(self.containers.rl_conf.savememorypath+'memoryTMP.pkl')-10240):
                        self.pload(self.containers.rl_conf.savememorypath+'memory.pkl', containers, lock)
                        print("Loading existing memory with", len(self._buffer), "entries", level=10)
                    else:
                        corrupted = True
                except:
                    corrupted = True
            if corrupted:
                print("Previous memory was corrupted!", level=10) 
                if os.path.exists(self.containers.rl_conf.savememorypath+'memoryTMP.pkl'):
                    if os.path.getsize(self.containers.rl_conf.savememorypath+'memoryTMP.pkl') > 1024: 
                        shutil.copyfile(self.containers.rl_conf.savememorypath+'memoryTMP.pkl', self.containers.rl_conf.savememorypath+'memory.pkl')
                        self.pload(self.containers.rl_conf.savememorypath+'memory.pkl', containers, lock)
                        print("Loading Backup-Memory with", len(self._buffer), "entries", level=10)
        
        
    def __len__(self):
        return self._size            


    
    def append(self,obj):
        oldstate, action, reward, newstate, fEnd = obj
        oldspeed = oldstate[1]
        oldstate = oldstate[0]
        newspeed = newstate[1]
        newstate = newstate[0]
        if self._pointer == 0:
            self._visionvecs[-self._state_stacksize:] = oldstate
            self._speeds[-1] = oldspeed
        
        self._visionvecs[self._pointer] = newstate[-1]
        self._speeds[self._pointer] = newspeed
        self._actions[self._pointer] = action
        self._rewards[self._pointer] = reward
        self._fEnds[self._pointer] = fEnd 
            
        self._pointer = (self._pointer+1)%self.capacity
        
        self._appendcount += 1
        if self._size < self.capacity:
            self._size += 1
        
    
    def __getitem__(self, index):
        #Get a list of (s,a,r,s',fE) tuples
                       
        action = self._actions[index]
        reward = self._rewards[index]
        speed = self._speeds[(index-1 % self.capacity)]
        folgespeed = self._speeds[index]
        fEnd = self._fEnds[index]
        if index-self._state_stacksize-1 >= 0:
            state = self._visionvecs[index-self._state_stacksize-1:index-1]
            folgestate = self._visionvecs[index-self._state_stacksize:index]
        else:
            state = [self._visionvecs[i] for i in ]
        
        state = (state, speed)
        folgestate = (folgestate, folgespeed)
                        
        return [state, action, reward, folgestate, fEnd]


    
    
    
    
#       THIS STUFF NEEDS TO BE DONE     
#    
#    def append(self, obj):
#        with self._lock:
#                        
#            self._buffer[self._pointer] = obj                      
#            self._pointer = (self._pointer + 1) % self.capacity            
#            
#                            
#            self._appendcount += 1
#            if self._size < self.capacity:
#                self._size += 1
#            
#            if self.containers.keep_memory: 
#                if ((current_milli_time() - self.lastsavetime) / (1000*60)) > self.containers.rl_conf.saveMemoryAllMins: #previously: if self._appendcount % self.containers.rl_conf.savememoryall == 0:
#                    self.save_memory()
#    
#    
#    def save_memory(self):
#        if self.containers.keep_memory: 
#            self.containers.myAgent.freezeEverything("saveMem")
#            self.psave(self.containers.rl_conf.savememorypath+'memoryTMP.pkl')
#            print("Saving Memory at",time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), level=6)
#            if os.path.exists(self.containers.rl_conf.savememorypath+'memoryTMP.pkl'):
#                if os.path.getsize(self.containers.rl_conf.savememorypath+'memoryTMP.pkl') > 1024: #only use it as memory if you weren't disturbed while writing
#                    shutil.copyfile(self.containers.rl_conf.savememorypath+'memoryTMP.pkl', self.containers.rl_conf.savememorypath+'memory.pkl')
#            self.lastsavetime = current_milli_time()
#            self.containers.myAgent.unFreezeEverything("saveMem")   
#           
#            
#            
#    def sample(self, n):
#        with self._lock:
#            return random.sample(self._buffer[:self._size-3], n) 
##            samples = np.random.permutation(self._size-4)[:n]
##            batch = [self._buffer[i] for i in samples]  
##            return batch                   
#                       
#        
#        
#    def pop(self):
#        self._pointer = (self._pointer - 1) % self.capacity
#        self._size -= 1
#        return self._buffer[self._pointer]
#        
#        
#    
#    def endEpisode(self):
#        lastmemoryentry = self.pop() #oldstate, action, reward, newstate, fEnd
#        if lastmemoryentry is not None:
#            lastmemoryentry[4] = True
#            self.append(lastmemoryentry)
#            
#            
#    def punishLastAction(self, howmuch):
#            lastmemoryentry = self.pop() #oldstate, action, reward, newstate, fEnd
#            if lastmemoryentry is not None:
#                lastmemoryentry[2] -= abs(howmuch)
#                self.append(lastmemoryentry)     
#    
#    
#    
#    
#    
#    
#    
#            
#        
#    #loads everything and then overwrites containers, locks, and lastsavetime, as those are pointers/relative to now.
#    def pload(self, filename, containers, lock):
#        with open(filename, 'rb') as f:
#            tmp_dict = pickle.load(f)
#        self.__dict__.update(tmp_dict) 
#        self.containers = containers
#        self._lock = lock
#        self.lastsavetime = current_milli_time()
#    
#    
#    def psave(self, filename):
#        odict = self.__dict__.copy() # copy the dict since we change it
#        del odict['containers']  
#        del odict['_lock']  
#        with open(filename, 'wb') as f:
#            pickle.dump(odict, f, pickle.HIGHEST_PROTOCOL)
