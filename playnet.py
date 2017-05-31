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
import tensorflow as tf

#====own classes====
import cnn
import read_supervised
from myprint import myprint as print


STANDARDRETURN = ("[0.5,0,0.0]", [0.5, 0, 0.0])


class PlayNet(object):
    def __init__(self, num, config, containers, rl_config_dummy=None, startfreshdummy=False): #der dummy ist da damit man playnet & reinfnet austauschen kan
        self.lock = threading.Lock()
        self.isinitialized = False
        self.containers = containers        
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
                returnstuff, original = self.performNetwork(othervecs, visionvec)
                self.containers.outputval.update(returnstuff, self.containers.inputval.timestamp)    
                self.isbusy = False
            finally:
                self.lock.release()
                  
                

    def performNetwork(self, othervecs, visionvec):
        print("Another ANN Inference")
        check, (networkresult, _) = self.cnn.run_inference(self.session, visionvec, othervecs, self.config.history_frame_nr)
        if check:
            throttle, brake, steer = read_supervised.dediscretize_all(networkresult[0], self.config.steering_steps, self.config.INCLUDE_ACCPLUSBREAK)
            result = "["+str(throttle)+", "+str(brake)+", "+str(steer)+"]"
            return result, [throttle, brake, steer]
        else:
            return STANDARDRETURN

            

    def initNetwork(self):
        with tf.Graph().as_default():    
            initializer = tf.random_uniform_initializer(-0.1, 0.1)
                                                 
            with tf.name_scope("runAsServ"):
                with tf.variable_scope("cnnmodel", reuse=None, initializer=initializer): 
                    self.cnn = cnn.CNN(self.config, is_reinforcement = False, is_training=False)
            
            print(self.cnn.trainvars)
            
            self.saver = tf.train.Saver(self.cnn.trainvars)
            self.session = tf.Session()
            ckpt = tf.train.get_checkpoint_state(self.config.checkpoint_dir) 
            assert ckpt and ckpt.model_checkpoint_path, "I need a supervisedly pre-trained net!"
            self.saver.restore(self.session, ckpt.model_checkpoint_path)
            print("network %s initialized" %str(self.number+1))
            self.isinitialized = True
