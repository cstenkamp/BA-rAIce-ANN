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
#====own classes====
from myprint import myprint as print


#we want something with fast random access, so we take a list instead of a deque https://wiki.python.org/moin/TimeComplexity
#https://docs.python.org/3/library/collections.html#deque-objects

SAVENAME = "memory"

#TODO: not sure how thread-safe this is.. https://stackoverflow.com/questions/13610654/how-to-make-built-in-containers-sets-dicts-lists-thread-safe
class Memory(object):
    def __init__(self, capacity, containers):
        self._lock = lock = threading.Lock()
        self.memorypath = self.containers.myAgent.folder(self.containers.rl_conf.memory_dir)
        self.capacity = capacity
        self._buffer = [None]*capacity #deque(elemtype, capacity)
        self._pointer = 0
        self._appendcount = 0
        self.containers = containers
        self.lastsavetime = current_milli_time()
        self._size = 0
        
        if self.containers.keep_memory:
            corrupted = False
            if os.path.exists(self.memorypath+SAVENAME+'.pkl'):
                try:
                    if os.path.getsize(self.memorypath+'.pkl') > 1024 and (os.path.getsize(self.memorypath+SAVENAME+'.pkl') >= os.path.getsize(self.memorypath+'TMP.pkl')-10240):
                        self.pload(self.memorypath+SAVENAME+'.pkl', containers, lock)
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
                        self.pload(self.memorypath+SAVENAME+'.pkl', containers, lock)
                        print("Loading Backup-Memory with", self._size, "entries", level=10)
        
        
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
            
            if self.containers.keep_memory: 
                if ((current_milli_time() - self.lastsavetime) / (1000*60)) > self.containers.rl_conf.saveMemoryAllMins: #previously: if self._appendcount % self.containers.rl_conf.savememoryall == 0:
                    self.save_memory()
    
    
    def save_memory(self):
        with self._lock:
            if self.containers.keep_memory: 
                self.containers.myAgent.freezeEverything("saveMem")
                self.psave(self.memorypath+SAVENAME+'TMP.pkl')
                print("Saving Memory at",time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), level=6)
                if os.path.exists(self.memorypath+SAVENAME+'TMP.pkl'):
                    if os.path.getsize(self.memorypath+SAVENAME+'TMP.pkl') > 1024: #only use it as memory if you weren't disturbed while writing
                        shutil.copyfile(self.memorypath+SAVENAME+'TMP.pkl', self.memorypath+SAVENAME+'.pkl')
                self.lastsavetime = current_milli_time()
                self.containers.myAgent.unFreezeEverything("saveMem")   
           
            
    def __getitem__(self, index):
        with self._lock:
            return self._buffer[index]
            
            
    def sample(self, n):
        with self._lock:
            return random.sample(self._buffer[:self._size], n) 
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

        
    def pop(self):
        self._pointer = (self._pointer - 1) % self.capacity
        self._size -= 1
        return self._buffer[self._pointer]
        
        
    
    def endEpisode(self):
        if self._size < 2:
            return
        lastmemoryentry = self.pop() #oldstate, action, reward, newstate, fEnd
        if lastmemoryentry is not None:
            lastmemoryentry[4] = True
            self.append(lastmemoryentry)
            
            
    def punishLastAction(self, howmuch):
        if self._size < 2:
            return
        lastmemoryentry = self.pop() #oldstate, action, reward, newstate, fEnd
        if lastmemoryentry is not None:
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




    @staticmethod
    def make_long_from_floats(acc, brk, steer):
        return acc, brk, steer
    
    @staticmethod
    def make_floats_from_long(content):
        return content[0], content[1], content[2]