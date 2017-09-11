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

SAVENAME = "Ememory"

#TODO: not sure how thread-safe this is.. https://stackoverflow.com/questions/13610654/how-to-make-built-in-containers-sets-dicts-lists-thread-safe
class Memory(object):
    def __init__(self, capacity, conf, agent, state_stacksize, constantmemorysize):
        self._lock = lock = threading.Lock()
        self.conf = conf
        self.agent = agent
        self.memorypath = self.agent.folder(self.conf.memory_dir)
        self.capacity = capacity
        self._state_stacksize = state_stacksize
        self._pointer = 0
        self._appendcount = 0
        self.lastsavetime = current_milli_time()
        self._size = 0
        
        if constantmemorysize: #sollte effizienter sein
            self._visionvecs = np.zeros((capacity+state_stacksize, self.conf.image_dims[0], self.conf.image_dims[1]), dtype=self.conf.visionvecdtype)
            if self.conf.use_second_camera:
                self._visionvecs2 = np.zeros((capacity+state_stacksize, self.conf.image_dims[0], self.conf.image_dims[1]), dtype=self.conf.visionvecdtype)
        else:
            self._visionvecs = [None]*(capacity+state_stacksize)
            if self.conf.use_second_camera:
                self._visionvecs2 = [None]*(capacity+state_stacksize)
        self._speeds = np.zeros(capacity+1, dtype=np.float) #da das state-speed von n+1 gleich dem folgestate-speed von n ist, muss er nur 1 mal doppelt abspeichern
        self._actions = np.zeros(capacity, dtype=np.uint32) #if I stored only the argmax, it could be np.int8
        self._rewards = np.zeros(capacity, dtype=np.float)
        self._fEnds = np.zeros(capacity, dtype=np.bool)
        #keine Folgestates, da die ja im n+1ten Element stecken
        
        
        if not self.agent.start_fresh:
            corrupted = False
            if os.path.exists(self.memorypath):
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
        
        
    def __len__(self):
        with self._lock:
            return self._size            
    
    
    def __getitem__(self, index): #if i had with self._lock here, I would get a deadlock in the sample-method. 
        
        if self._appendcount > self.capacity and (self._pointer <= index <= self._pointer+3): #I know that the values from _pointer to _pointer+3 are always wrong.
            return False
        if index >= self._size:
            return None
        
        action = self.make_floats_from_long(self._actions[index])
        actHist = [self.make_floats_from_long(self._actions[(i % self.capacity)]) for i in range(index,index-4,-1)]
        newest = self.make_floats_from_long(self._actions[((index+1) % self.capacity)]) 
        newest = None if newest == (0, 0, 0) else newest
        
        reward = self._rewards[index]
        speed = self._speeds[index]
        folgespeed = self._speeds[(index+1 % self.capacity)]
        fEnd = self._fEnds[index]
        state = list(reversed(self._visionvecs[index:index+4]))
        folgestate = list(reversed(self._visionvecs[index+1:index+5]))
        if self.conf.use_second_camera:
            state2 = list(reversed(self._visionvecs2[index:index+4]))
            folgestate2 = list(reversed(self._visionvecs2[index+1:index+5]))
                
        
        for j in range(1, self._state_stacksize):
            if self._fEnds[(index-j % self.capacity)]:
                iter1 = True
                for i in range(j, self._state_stacksize):
                    state[i] = np.zeros(state[i].shape)
                    if self.conf.use_second_camera:
                        state2[i] = np.zeros(state2[i].shape)
                    if not iter1: 
                        folgestate[i] = np.zeros(folgestate[i].shape)
                        if self.conf.use_second_camera:
                            folgestate2[i] = np.zeros(folgestate2[i].shape)
                    iter1 = False
            
        if self.conf.use_second_camera:    
            state = np.concatenate([state, state2])
            folgestate = np.concatenate([folgestate, folgestate2])

        state = (state, [speed]+actHist)
        folgestate = (folgestate, [folgespeed]+[newest]+actHist[:3])
                        
        return [state, action, reward, folgestate, fEnd]


    
    def append(self,obj):
        with self._lock:
            oldstate, action, reward, newstate, fEnd = obj
            action = self.make_long_from_floats(*action)
            oldspeed = oldstate[1][0]
            oldstate = oldstate[0]
            newspeed = newstate[1][0]
            newstate = newstate[0]
            if self.conf.use_second_camera:   
                oldstat2 = oldstate[oldstate.shape[0]//2:,:,:]
                oldstate = oldstate[:oldstate.shape[0]//2,:,:]
                newstat2 = newstate[newstate.shape[0]//2:,:,:]
                newstate = newstate[:newstate.shape[0]//2,:,:]
                
            if self._pointer == 0:
                self._visionvecs[0:self._state_stacksize] = np.array(list(reversed(oldstate)), dtype=self.conf.visionvecdtype)
                self._visionvecs[self._state_stacksize] = np.array(newstate[0], dtype=self.conf.visionvecdtype)
                if self.conf.use_second_camera:   
                    self._visionvecs2[0:self._state_stacksize] = np.array(list(reversed(oldstat2)), dtype=self.conf.visionvecdtype)
                    self._visionvecs2[self._state_stacksize] = np.array(newstat2[0], dtype=self.conf.visionvecdtype)
                self._speeds[0] = oldspeed
            else:
                self._visionvecs[self._pointer+self._state_stacksize] = np.array(newstate[0], dtype=self.conf.visionvecdtype)
                if self.conf.use_second_camera:   
                    self._visionvecs2[self._pointer+self._state_stacksize] = np.array(newstat2[0], dtype=self.conf.visionvecdtype)
            
            if self._fEnds[(self._pointer-1 % self.capacity)]:                                          #if he resettet last the last frame, Q-learning doesn't look at its s' anyway...
                self._visionvecs[(self._pointer+self._state_stacksize-1 % self.capacity)] = np.array(oldstate[0], dtype=self.conf.visionvecdtype) #but we need it this time, because we skipped exactly one frame ("if oldstate is not None" in agents addtomemory)
                if self.conf.use_second_camera:   
                    self._visionvecs2[(self._pointer+self._state_stacksize-1 % self.capacity)] = np.array(oldstat2[0], dtype=self.conf.visionvecdtype)  
                self._speeds[self._pointer] = oldspeed                 
                
            self._speeds[self._pointer+1] = newspeed
            self._actions[self._pointer] = action
            self._rewards[self._pointer] = reward
            self._fEnds[self._pointer] = fEnd 
                       
            self._pointer = (self._pointer+1) % self.capacity
            
            self._appendcount += 1
            if self._size < self.capacity:
                self._size += 1
        
            if self.agent.keep_memory and self.conf.save_memory_all_mins: 
                if ((current_milli_time() - self.lastsavetime) / (1000*60)) > self.conf.save_memory_all_mins: 
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
           
            
            
    def sample(self, n): #gesamplet wird nur sobald len(self.memory) > self.conf.batch_size+self.conf.history_frame_nr+1
        with self._lock:
            assert self._size > self._state_stacksize, "you can't even sample a single value!"
            if self._appendcount <= self.capacity:
                samples = list(np.random.permutation(self._size)[:n])
            else:
                samples = np.random.permutation(self._size-self._state_stacksize)[:n] 
                samples = [i if i < self._pointer else i+self._state_stacksize for i in samples ]
                #because again,  I know that the values from _pointer to _pointer+3 are always wrong. So I simply don't use them, its 4 out of thousands of values.
            
            batch = [self[i] for i in samples]
            
            return batch
        
                       
        
        
    def _pop(self):
        tmp = (self._pointer - 1) % self.capacity
        tmp2 = self[tmp]
        self._size -= 1
        self._pointer = tmp 
        return tmp2
        
        
    
    def endEpisode(self):
        if self._size < 2:
            return
        lastmemoryentry = self._pop() #oldstate, action, reward, newstate, fEnd
        if lastmemoryentry is not None and lastmemoryentry != False:
            lastmemoryentry[4] = True
            self.append(lastmemoryentry)
            
            
    def punishLastAction(self, howmuch):
        if self._size < 2:
            return
        lastmemoryentry = self._pop() #oldstate, action, reward, newstate, fEnd
        if lastmemoryentry is not None and lastmemoryentry != False:
            lastmemoryentry[2] -= abs(howmuch)
            self.append(lastmemoryentry)     
   

 
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
        del odict['agent']
        del odict['_lock']  
        with open(filename, 'wb') as f:
            pickle.dump(odict, f, pickle.HIGHEST_PROTOCOL)



    #I changed from storing the action as argmax to storing the action as (throttle, brake, steer)... this is necessary for that to still be an int
    @staticmethod
    def make_long_from_floats(throttle, brake, steer):
        ACCURACY = 2 #ab accuracy=3 müssste man uint64 nehmen
        throttle = round(throttle, ACCURACY)*10**(ACCURACY*3+2)
        brake = round(brake, ACCURACY)*10**(ACCURACY*2+1)
        steer = round((1+steer), ACCURACY)*10**(ACCURACY*1)
        return throttle+brake+steer
    
    
    @staticmethod
    def make_floats_from_long(value):
        ACCURACY = 2 
        throttle = round(value / (10**(ACCURACY*3+2)), ACCURACY)
        brake = round((value % 10**(ACCURACY*2+2))/ (10**(ACCURACY*2+1)), ACCURACY)
        steer = round((value % 10**(ACCURACY*1+1))/ (10**(ACCURACY*1)) -1, ACCURACY)
        return throttle, brake, steer
        