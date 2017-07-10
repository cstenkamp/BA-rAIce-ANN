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
        
        self._visionvecs = [None]*(capacity+state_stacksize)
        self._speeds = [None]*(capacity+1) #da das state-speed von n+1 gleich dem folgestate-speed von n ist, muss er nur 1 mal doppelt abspeichern
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
        with self._lock:
            return self._size            
    
    
    def __getitem__(self, index): #if i had with self._lock here, I would get a deadlock in the sample-method. 
        #Get a (s,a,r,s',fE) tuple  #TODO: Alex' Version... https://github.com/ahoereth/ddpg/blob/feature/rewrite/src/lib/memory.py#L28-L60. is it more efficient?
        
        if self._appendcount > self.capacity and (self._pointer <= index <= self._pointer+3): #I know that the values from _pointer to _pointer+3 are always wrong.
            return False
        if index >= self._size:
            return None
        
        action = self._actions[index]
        reward = self._rewards[index]
        speed = self._speeds[index]
        folgespeed = self._speeds[(index+1 % self.capacity)]
        fEnd = self._fEnds[index]
        state = self._visionvecs[index:index+4]
        folgestate = self._visionvecs[index+1:index+5]
        
        state = (state, speed)
        folgestate = (folgestate, folgespeed)
                        
        return [state, action, reward, folgestate, fEnd]


    
    def append(self,obj):
        with self._lock:
            oldstate, action, reward, newstate, fEnd = obj
            oldspeed = oldstate[1]
            oldstate = oldstate[0]
            newspeed = newstate[1]
            newstate = newstate[0]
            if self._pointer == 0:
                self._visionvecs[0:self._state_stacksize] = oldstate
                self._visionvecs[self._state_stacksize] = newstate[-1]
                self._speeds[0] = oldspeed
            else:
                self._visionvecs[self._pointer+self._state_stacksize] = newstate[-1]
                
            self._speeds[self._pointer+1] = newspeed
            self._actions[self._pointer] = action
            self._rewards[self._pointer] = reward
            self._fEnds[self._pointer] = fEnd 
                
            self._pointer = (self._pointer+1) % self.capacity
            
            self._appendcount += 1
            if self._size < self.capacity:
                self._size += 1
        
            if self.containers.keep_memory: 
                if ((current_milli_time() - self.lastsavetime) / (1000*60)) > self.containers.rl_conf.saveMemoryAllMins: 
                    self.save_memory()
    
    
    
    def save_memory(self):
        with self._lock:
            if self.containers.keep_memory: 
                self.containers.myAgent.freezeEverything("saveMem")
                self.psave(self.containers.rl_conf.savememorypath+'memoryTMP.pkl')
                print("Saving Memory at",time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), level=6)
                if os.path.exists(self.containers.rl_conf.savememorypath+'memoryTMP.pkl'):
                    if os.path.getsize(self.containers.rl_conf.savememorypath+'memoryTMP.pkl') > 1024: #only use it as memory if you weren't disturbed while writing
                        shutil.copyfile(self.containers.rl_conf.savememorypath+'memoryTMP.pkl', self.containers.rl_conf.savememorypath+'memory.pkl')
                self.lastsavetime = current_milli_time()
                self.containers.myAgent.unFreezeEverything("saveMem")   
           
            
            
    def sample(self, n): #gesamplet wird nur sobald len(self.memory) > self.rl_config.batchsize+self.rl_config.history_frame_nr+1
        with self._lock:
            assert self._size > self._state_stacksize, "you can't even sample a single value!"
            if self._appendcount <= self.capacity:
                samples = list(np.random.permutation(self._size)[:n])
            else:
                samples = np.random.permutation(self._size-self._state_stacksize)[:n] 
                samples = [i if i < self._pointer else i+self._state_stacksize for i in samples ]
                #because again,  I know that the values from _pointer to _pointer+3 are always wrong. So I simply don't use them, its 4 out of thousands of values.
            
            batch = [self[i] for i in samples]
            
            return samples, batch
        
                       
        
        
    def pop(self):
        tmp = (self._pointer - 1) % self.capacity
        tmp2 = self[tmp]
        self._size -= 1
        self._pointer = tmp 
        return tmp2
        
        
    
    def endEpisode(self):
        if self._size < 2:
            return
        lastmemoryentry = self.pop() #oldstate, action, reward, newstate, fEnd
        if lastmemoryentry is not None and lastmemoryentry != False:
            lastmemoryentry[4] = True
            self.append(lastmemoryentry)
            
            
    def punishLastAction(self, howmuch):
        if self._size < 2:
            return
        lastmemoryentry = self.pop() #oldstate, action, reward, newstate, fEnd
        if lastmemoryentry is not None and lastmemoryentry != False:
            lastmemoryentry[2] -= abs(howmuch)
            self.append(lastmemoryentry)     
   

 
    #loads everything and then overwrites containers, locks, and lastsavetime, as those are pointers/relative to now.
    def pload(self, filename, containers, lock):
        with open(filename, 'rb') as f:
            tmp_dict = pickle.load(f)
        self.__dict__.update(tmp_dict) 
        self.containers = containers
        self._lock = lock
        self.lastsavetime = current_milli_time()
    
    
    def psave(self, filename):
        odict = self.__dict__.copy() # copy the dict since we change it
        del odict['containers']  
        del odict['_lock']  
        with open(filename, 'wb') as f:
            pickle.dump(odict, f, pickle.HIGHEST_PROTOCOL)
