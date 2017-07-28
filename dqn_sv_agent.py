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



class DQN_SV_Agent(AbstractAgent):
    def __init__(self, config, containers, rl_config_dummy=None, startfresh_dummy=False, *args, **kwargs): #der dummy ist da damit man playnet & reinfnet austauschen kan
        super().__init__(config, containers, *args, **kwargs)
        self.ff_inputsize = 30
        self.network = dqn.CNN


    def runInference(self, gameState, pastState): #since we don't have a memory in this agent, we don't care for other_inputs_toSave
        if self.isinitialized and self.checkIfInference():
            conv_inputs, other_inputs = self.getAgentState(*gameState)
            super().preRunInference()
                
            self.lock.acquire()
            try:
                self.isbusy = True 
                toUse, toSave = self.performNetwork(conv_inputs, self.makeNetUsableOtherInputs(other_inputs))
                super().postRunInference(toUse, toSave)
                self.isbusy = False
            finally:
                self.lock.release()
                  
                

    def performNetwork(self, conv_inputs, inflated_other_inputs):
        super().performNetwork(conv_inputs, inflated_other_inputs)
        networkresult, _ = self.cnn.run_inference(self.session, conv_inputs, inflated_other_inputs) 
        throttle, brake, steer = self.dediscretize(networkresult[0], self.sv_conf)
        toUse = "["+str(throttle)+", "+str(brake)+", "+str(steer)+"]"
        return toUse, (throttle, brake, steer)

            

    def initNetwork(self):        
        initializer = tf.random_uniform_initializer(-0.1, 0.1)
                                             
        with tf.name_scope("runAsServ"):
            with tf.variable_scope("cnnmodel", reuse=None, initializer=initializer): 
                self.cnn = self.network(self.sv_conf, self, mode="inference")
        
        print(self.cnn.trainvars)
        
        self.saver = tf.train.Saver(self.cnn.trainvars)
        self.session = tf.Session()
        ckpt = tf.train.get_checkpoint_state(self.sv_conf.checkpoint_dir) 
        assert ckpt and ckpt.model_checkpoint_path, "I need a supervisedly pre-trained net!"
        self.saver.restore(self.session, ckpt.model_checkpoint_path)
        print("network initialized")
        self.isinitialized = True
            
            
###############################################################################
if __name__ == '__main__':  
    import config
    conf = config.Config()
    agent = DQN_SV_Agent(conf, None)
    agent.svTrain()