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

import tensorflow as tf

#====own classes====
from agent import AbstractAgent
import dqn
from myprint import myprint as print


STANDARDRETURN = ("[0.5,0,0.0]", [0.5, 0, 0.0])


class PlayNetAgent(AbstractAgent):
    def __init__(self, config, containers, rl_config_dummy=None, startfreshdummy=False, *args, **kwargs): #der dummy ist da damit man playnet & reinfnet austauschen kan
        super().__init__(containers, *args, **kwargs)
        self.config = config
        self.initNetwork()

    
    def runInference(self, update_only_if_new):
        if self.isinitialized:
            super().runInference(update_only_if_new)
                
            self.lock.acquire()
            try:
                self.isbusy = True 
                otherinputs, visionvec = self.containers.inputval.read()
                returnstuff, original = self.performNetwork(otherinputs, visionvec)
                self.containers.outputval.update(returnstuff, self.containers.inputval.CTimestamp, self.containers.inputval.STimestamp)    
                self.isbusy = False
            finally:
                self.lock.release()
                  
                

    def performNetwork(self, otherinputs, visionvec):
        super().performNetwork(otherinputs, visionvec)
        
        check, (networkresult, _) = self.cnn.run_inference(self.session, visionvec, otherinputs, self.config.history_frame_nr)
        if check:
            throttle, brake, steer = self.dediscretize(networkresult[0], self.config)
            result = "["+str(throttle)+", "+str(brake)+", "+str(steer)+"]"
            return result, [throttle, brake, steer]
        else:
            return STANDARDRETURN

            

    def initNetwork(self):
        with tf.Graph().as_default():    
            initializer = tf.random_uniform_initializer(-0.1, 0.1)
                                                 
            with tf.name_scope("runAsServ"):
                with tf.variable_scope("cnnmodel", reuse=None, initializer=initializer): 
                    self.cnn = dqn.CNN(self.config, mode="inference")
            
            print(self.cnn.trainvars)
            
            self.saver = tf.train.Saver(self.cnn.trainvars)
            self.session = tf.Session()
            ckpt = tf.train.get_checkpoint_state(self.config.checkpoint_dir) 
            assert ckpt and ckpt.model_checkpoint_path, "I need a supervisedly pre-trained net!"
            self.saver.restore(self.session, ckpt.model_checkpoint_path)
            print("network initialized")
            self.isinitialized = True
