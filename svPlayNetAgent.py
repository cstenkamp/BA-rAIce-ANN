# -*- coding: utf-8 -*-
"""
Created on Wed May 10 13:33:46 2017

@author: nivradmin
"""

import tensorflow as tf

#====own classes====
from agent import AbstractAgent
import dqn
from myprint import myprint as print



class PlayNetAgent(AbstractAgent):
    def __init__(self, config, containers, rl_config_dummy=None, startfresh_dummy=False, *args, **kwargs): #der dummy ist da damit man playnet & reinfnet austauschen kan
        super().__init__(config, containers, *args, **kwargs)
        self.initNetwork()


    def runInference(self, conv_inputs, other_inputs, _): #since we don't have a memory in this agent, we don't care for other_inputs_toSave
        if self.isinitialized and self.checkIfInference():
            super().preRunInference(None)
                
            self.lock.acquire()
            try:
                self.isbusy = True 
                toUse, toSave = self.performNetwork(other_inputs, conv_inputs)
                super().postRunInference(toUse, toSave)
                self.isbusy = False
            finally:
                self.lock.release()
                  
                

    def performNetwork(self, other_inputs, conv_inputs):
        super().performNetwork(other_inputs, conv_inputs)
        networkresult, _ = self.cnn.run_inference(self.session, conv_inputs, other_inputs) 
        throttle, brake, steer = self.dediscretize(networkresult[0], self.sv_conf)
        toUse = "["+str(throttle)+", "+str(brake)+", "+str(steer)+"]"
        return toUse, (throttle, brake, steer)

            

    def initNetwork(self):
        with tf.Graph().as_default():    
            initializer = tf.random_uniform_initializer(-0.1, 0.1)
                                                 
            with tf.name_scope("runAsServ"):
                with tf.variable_scope("cnnmodel", reuse=None, initializer=initializer): 
                    self.cnn = dqn.CNN(self.sv_conf, mode="inference")
            
            print(self.cnn.trainvars)
            
            self.saver = tf.train.Saver(self.cnn.trainvars)
            self.session = tf.Session()
            ckpt = tf.train.get_checkpoint_state(self.sv_conf.checkpoint_dir) 
            assert ckpt and ckpt.model_checkpoint_path, "I need a supervisedly pre-trained net!"
            self.saver.restore(self.session, ckpt.model_checkpoint_path)
            print("network initialized")
            self.isinitialized = True
