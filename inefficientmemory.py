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

SAVENAME = "memory"

#TODO: not sure how thread-safe this is.. https://stackoverflow.com/questions/13610654/how-to-make-built-in-containers-sets-dicts-lists-thread-safe
class Memory(object):
    def __init__(self, capacity, conf, agent, load=True):
        self._lock = lock = threading.Lock()
        self.conf = conf
        self.agent = agent
        self.memorypath = self.agent.folder(self.conf.memory_dir)
        self.capacity = capacity
        self._buffer = [None]*capacity #deque(elemtype, capacity)
        self._pointer = 0
        self._appendcount = 0
        self.lastsavetime = current_milli_time()
        self._size = 0
        
        if (not self.agent.start_fresh) and load:
            corrupted = False
            if os.path.exists(self.memorypath+SAVENAME+'.pkl'):
                try:
                    if os.path.getsize(self.memorypath+SAVENAME+'.pkl') > 1024 and (os.path.getsize(self.memorypath+SAVENAME+'.pkl') >= os.path.getsize(self.memorypath+SAVENAME+'TMP.pkl')-10240):
                        self.pload(self.memorypath+SAVENAME+'.pkl', conf, agent, lock)
                        print("Loading existing memory with", self._size, "entries", level=10)
                    else:
                        corrupted = True
                except:
                    corrupted = True
            if corrupted:
                print("Previous memory was corrupted!", level=10) 
                if os.path.exists(self.memorypath+SAVENAME+'TMP.pkl'):
                    if os.path.getsize(self.memorypath+SAVENAME+'TMP.pkl') > 1024: 
                        shutil.copyfile(self.memorypath+SAVENAME+'TMP.pkl', self.memorypath+SAVENAME+'.pkl')
                        self.pload(self.memorypath+SAVENAME+'.pkl', conf, agent, lock)
                        print("Loading Backup-Memory with", self._size, "entries", level=10)
            
        self.epistart = self._pointer
        
        
    def __len__(self):
        with self._lock:
            return self._size            
            
    
    def append(self, obj):
        with self._lock:
                        
            self._buffer[self._pointer] = obj                      
            self._pointer = (self._pointer + 1) % self.capacity            
            
            self._appendcount += 1
            if self._size < self.capacity:
                self._size += 1
            
            if self.agent.keep_memory and self.conf.save_memory_all_mins: 
                if ((current_milli_time() - self.lastsavetime) / (1000*60)) > self.conf.save_memory_all_mins: #previously: if self._appendcount % self.conf.savememoryall == 0:
                    self.save_memory()
    
    
    def save_memory(self):
        with self._lock:
            if self.agent.keep_memory: 
                self.agent.freezeEverything("saveMem")
                self.psave(self.memorypath+SAVENAME+'TMP.pkl')
                print("Saving Memory at",time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), level=6)
                if os.path.exists(self.memorypath+SAVENAME+'TMP.pkl'):
                    if os.path.getsize(self.memorypath+SAVENAME+'TMP.pkl') > 1024: #only use it as memory if you weren't disturbed while writing
                        shutil.copyfile(self.memorypath+SAVENAME+'TMP.pkl', self.memorypath+SAVENAME+'.pkl')
                self.lastsavetime = current_milli_time()
                self.agent.unFreezeEverything("saveMem")   
           
            
    def __getitem__(self, index):
        with self._lock:
            return self._buffer[index]
            
            
    def sample(self, n):
        with self._lock:
            return random.sample(self._buffer[:self._size], n) 
            #returns [[s,a,r,s2,t],[s,a,r,s2,t],[s,a,r,s2,t],...]
#            samples = np.random.permutation(self._size-4)[:n]
#            batch = [self._buffer[i] for i in samples]  
#            return batch                   
#                  
#    def sample2(self, n):
#        return zip(*self.sample(n))      
#  
#    
#    def sampletest(self, samples):
#        batch = [self._buffer[i] for i in samples]  
#        return batch            

        
    def _pop(self):
        self._pointer = (self._pointer - 1) % self.capacity
        self._size -= 1
        return self._buffer[self._pointer]
        
        
    
    def endEpisode(self):
        if self._size < 2:
            return
        
        lastmemoryentry = self._pop() #oldstate, action, reward, newstate, fEnd
        if lastmemoryentry is not None:
            lastmemoryentry[4] = True
            self.append(lastmemoryentry)
#        for j in range(self._pointer-10, self._pointer):
#            print(self._buffer[j][0][1][0], self.agent.dediscretize(self.agent.discretize(*self._buffer[j][1])), self._buffer[j][2], self._buffer[j][3][1][0], self._buffer[j][4])
        prev_epistart = self.epistart
        self.epistart = self._pointer
        return prev_epistart, self.epistart
            
        
    def punishLastAction(self, howmuch):
        if self._size < 2:
            return
        lastmemoryentry = self._pop() #oldstate, action, reward, newstate, fEnd
        if lastmemoryentry is not None:
            lastmemoryentry[2] -= abs(howmuch)
            self.append(lastmemoryentry)     
    
    
    def average_rewards(self, fromwhere, towhere):
        if fromwhere < towhere:
            tmp = self._buffer[fromwhere:towhere]
        elif fromwhere > towhere:
            tmp = self._buffer[fromwhere:]+self._buffer[:towhere]
        else:
            tmp = [self._buffer[fromwhere]]
            
        rewards = np.array(list(zip(*tmp))[2])
        return np.mean(rewards)
    
    
    
    
            
        
    #loads everything and then overwrites conf, locks, and lastsavetime, as those are pointers/relative to now.
    def pload(self, filename, conf, agent, lock):
        with open(filename, 'rb') as f:
            tmp_dict = pickle.load(f)
        self.__dict__.update(tmp_dict) 
        self.conf = conf
        self.agent = agent
        self._lock = lock
        self.lastsavetime = current_milli_time()
    
    
    def psave(self, filename):
        odict = self.__dict__.copy() # copy the dict since we change it
        del odict['conf']  
        del odict['_lock']  
        del odict['agent']
        with open(filename, 'wb') as f:
            pickle.dump(odict, f, pickle.HIGHEST_PROTOCOL)
