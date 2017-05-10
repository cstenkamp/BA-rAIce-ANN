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

#====own classes====
import supervisedcnn 
import read_supervised


STANDARDRETURN = "[0,0,0.0]"




class PlayNet(object):
    def __init__(self, num, config):
        self.lock = threading.Lock()
        self.isinitialized = False
        self.containers = None        
        self.number = num
        self.isbusy = False
        tps = read_supervised.TPList(read_supervised.FOLDERNAME, config.msperframe)
        self.config = config
        self.normalizers = tps.find_normalizers()
        self.initNetwork()

    @staticmethod
    def flatten_oneDs(AllOneDs):
        return np.array(read_supervised.flatten(AllOneDs))
    
    @staticmethod
    def normalize_oneDs(FlatOneDs, normalizers):
        FlatOneDs -= np.array([item[0] for item in normalizers])
        NormalizedOneDs = FlatOneDs / np.array([item[1] for item in normalizers])
        return NormalizedOneDs
        
    
    def runANN(self, update_only_if_new):
        if self.isinitialized:
            if update_only_if_new and self.containers.inputval.alreadyread:
                    return
                
            self.lock.acquire()
            try:
                self.isbusy = True
                print("Another ANN Starts")  
#                if self.containers.inputval.othervecs[0][0] > 30 and self.containers.inputval.othervecs[0][0] < 40:
#                    self.containers.outputval.send_via_senderthread("pleasereset", self.containers.inputval.timestamp)
#                    return
                returnstuff = self.performNetwork(self.containers.inputval)
                self.containers.outputval.update(returnstuff, self.containers.inputval.timestamp)    
                self.isbusy = False
            finally:
                self.lock.release()
                

    def performNetwork(self, inputval):
        _, visionvec = inputval.read()
        check, networkresult = self.cnn.run_inference(self.session, visionvec, self.config.history_frame_nr)
        if check:
            throttle, brake, steer = read_supervised.dediscretize_all((networkresult)[0])
            result = "["+str(throttle)+", "+str(brake)+", "+str(steer)+"]"
            return result
        else:
            return STANDARDRETURN


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
