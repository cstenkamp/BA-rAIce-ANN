# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 16:16:35 2017

@author: csten_000
"""


import os
from collections import deque
import pickle
import threading
import shutil
import time
current_milli_time = lambda: int(round(time.time() * 1000))


#wenn ich hier thread-locks verwenden würde würde er jedes mal einen neuen receiver-thread starten. 
#TODO: not sure how thread-safe this is.. https://stackoverflow.com/questions/13610654/how-to-make-built-in-containers-sets-dicts-lists-thread-safe
class Memory(object):
    def __init__(self, elemtype, size, containers):
        self._lock = threading.Lock()
        self.memory = deque(elemtype, size)
        self.appendcount = 0
        self.containers = containers
        self.lastsavetime = current_milli_time()
        if self.containers.keep_memory:
            corrupted = False
            if os.path.exists(self.containers.rl_conf.savememorypath+'memory.pkl'):
                try:
                    if os.path.getsize(self.containers.rl_conf.savememorypath+'memory.pkl') > 1024:
                        with open(self.containers.rl_conf.savememorypath+'memory.pkl', 'rb') as input:
                            self.memory = pickle.load(input)   
                        print("Loading existing memory with", len(self.memory), "entries", level=10)
                    else:
                        corrupted = True
                except:
                    corrupted = True
            if corrupted:
                print("Previous memory was corrupted!", level=10) 
                if os.path.exists(self.containers.rl_conf.savememorypath+'memoryTMP.pkl'):
                    if os.path.getsize(self.containers.rl_conf.savememorypath+'memoryTMP.pkl') > 1024: 
                        shutil.copyfile(self.containers.rl_conf.savememorypath+'memoryTMP.pkl', self.containers.rl_conf.savememorypath+'memory.pkl')
                        with open(self.containers.rl_conf.savememorypath+'memory.pkl', 'rb') as input:
                            self.memory = pickle.load(input)   
                        print("Loading Backup-Memory with", len(self.memory), "entries", level=10)
                
    
    def append(self, obj):
        with self._lock:
            self.memory.append(obj)
            self.appendcount += 1
            if self.containers.keep_memory: #TODO: sollte der das vielleicht in nem thread machen, damit der nicht zwischendurch unterbrochen wird?
                #previously: if self.appendcount % self.containers.rl_conf.savememoryall == 0:
                if ((current_milli_time() - self.lastsavetime) / (1000*60)) > self.containers.rl_conf.saveMemoryAllMins:
                    self.containers.myAgent.freezeEverything()
                    with open(self.containers.rl_conf.savememorypath+'memoryTMP.pkl', 'wb') as output:
                        pickle.dump(self.memory, output, pickle.HIGHEST_PROTOCOL)
                    print("Saving Memory at",time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),level=6)
                    if os.path.exists(self.containers.rl_conf.savememorypath+'memoryTMP.pkl'):
                        if os.path.getsize(self.containers.rl_conf.savememorypath+'memoryTMP.pkl') > 1024: #only use it as memory if you weren't disturbed while writing
                            shutil.copyfile(self.containers.rl_conf.savememorypath+'memoryTMP.pkl', self.containers.rl_conf.savememorypath+'memory.pkl')
                    self.lastsavetime = current_milli_time()
                    self.containers.myAgent.unFreezeEverything()   
                    
                    
    def pop(self):
        with self._lock:
            try:
                return self.memory.pop()        
            except:
                return None
