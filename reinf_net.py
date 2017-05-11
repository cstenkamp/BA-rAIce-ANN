# -*- coding: utf-8 -*-
"""
Created on Wed May 10 13:33:46 2017

@author: nivradmin
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 10 13:31:54 2017

@author: nivradmin
"""

import threading
import numpy as np
import tensorflow as tf
from collections import deque

#====own classes====
import supervisedcnn 
import read_supervised
import server


STANDARDRETURN = ("[0.5,0,0.0]", [0.5, 0, 0.0])
MEMORY_SIZE = 5000
epsilon = 0.001

#TODO: das memory muss auch ein gemeinsames object aller nets sein, sonst kann ich nicht mehr als 1 haben!

class ReinfNet(object):
    def __init__(self, num, config):
        self.lock = threading.Lock()
        self.isinitialized = False
        self.containers = None        
        self.number = num
        self.isbusy = False
        self.config = config
        #tps = read_supervised.TPList(read_supervised.FOLDERNAME, config.msperframe)
        #self.normalizers = tps.find_normalizers()
        self.initNetwork()

#    @staticmethod
#    def flatten_oneDs(AllOneDs):
#        return np.array(read_supervised.flatten(AllOneDs))
#    
#    @staticmethod
#    def normalize_oneDs(FlatOneDs, normalizers):
#        FlatOneDs -= np.array([item[0] for item in normalizers])
#        NormalizedOneDs = FlatOneDs / np.array([item[1] for item in normalizers])
#        return NormalizedOneDs
        
    
    def runANN(self, update_only_if_new):
        if self.isinitialized:
            if update_only_if_new and self.containers.inputval.alreadyread:
                    return
                
            self.lock.acquire()
            try:
                self.isbusy = True 
#                if self.containers.inputval.othervecs[0][0] > 30 and self.containers.inputval.othervecs[0][0] < 40:
#                    self.containers.outputval.send_via_senderthread("pleasereset", self.containers.inputval.timestamp)
#                    return
                othervecs, visionvec = self.containers.inputval.read()
                
                #add to memory
                oldstate, action = self.containers.inputval.get_previous_state()
                if oldstate is not None:
                    newstate = (visionvec, othervecs[1][4])
                    reward = self.calculateReward()
                    self.containers.memory.append([oldstate, action, reward, newstate]) 
                

                #run ANN
                if np.random.random() > epsilon:
                    returnstuff, original = self.performNetwork(othervecs, visionvec)
                else:
                    returnstuff, original = self.randomAction()
                    
                self.containers.inputval.addResultAndBackup(original) 
                self.containers.outputval.update(returnstuff, self.containers.inputval.timestamp)    
                self.isbusy = False
            finally:
                self.lock.release()
                
                
    def calculateReward(self):
        progress_old = self.containers.inputval.previous_othervecs[0][0]
        progress_new = self.containers.inputval.othervecs[0][0]
        if progress_old > 90 and progress_new < 10:
            progress_new += 100
        progress = round(progress_new-progress_old,3)
        
        stay_on_street = abs(self.containers.inputval.othervecs[3][0])
        stay_on_street = round(0 if stay_on_street < 5 else 100 if stay_on_street > 10 else stay_on_street-5, 3)
        
        
        return progress-stay_on_street

        
                        
                

    def performNetwork(self, othervecs, visionvec):
        print("Another ANN Inference")
        check, networkresult = self.cnn.run_inference(self.session, visionvec, othervecs, self.config.history_frame_nr)
        if check:
            throttle, brake, steer = read_supervised.dediscretize_all((networkresult)[0])
            result = "["+str(throttle)+", "+str(brake)+", "+str(steer)+"]"
            return result, [throttle, brake, steer]
        else:
            return STANDARDRETURN

            
    def randomAction(self):
        print("Random Action!")
        throttle = 1 if np.random.random() > 0.5 else 0
        brake = 1 if np.random.random() > 0.5 else 0
        steer = ((np.random.random()*2)-1)
        result = "["+str(throttle)+", "+str(brake)+", "+str(steer)+"]"
        return result, [throttle, brake, steer]
              
            

    def initNetwork(self):
        with tf.Graph().as_default():    
            initializer = tf.random_uniform_initializer(-0.1, 0.1)
                                                 
            with tf.name_scope("runAsServ"):
                with tf.variable_scope("cnnmodel", reuse=None, initializer=initializer): 
                    self.cnn = supervisedcnn.CNN(self.config, is_training=False)
            
            self.saver = tf.train.Saver(self.cnn.trainvars)
            self.session = tf.Session()
            ckpt = tf.train.get_checkpoint_state(self.config.checkpoint_dir) 
            assert ckpt and ckpt.model_checkpoint_path
            self.saver.restore(self.session, ckpt.model_checkpoint_path)
            print("network %s initialized" %str(self.number+1))
            self.isinitialized = True
