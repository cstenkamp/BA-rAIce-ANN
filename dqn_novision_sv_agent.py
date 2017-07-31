# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 12:41:01 2017

@author: csten_000
"""

import tensorflow as tf
import numpy as np

#====own classes====
from agent import AbstractAgent
from myprint import myprint as print



class Agent(AbstractAgent):
    def __init__(self, config, containers, rl_config_dummy=None, startfresh_dummy=False, *args, **kwargs): #der dummy ist da damit man playnet & reinfnet austauschen kan
        self.name = __file__[__file__.rfind("\\")+1:__file__.rfind(".")]
        super().__init__(config, containers, *args, **kwargs)
        self.ff_inputsize = 49
        self.usesConv = False
        self.ff_stacked = True
        
    #Override
    def getAgentState(self, vvec1_hist, vvec2_hist, otherinput_hist, action_hist):  
        other_inputs = np.ravel([i.returnRelevant() for i in otherinput_hist])
        stands_inputs = otherinput_hist[0].SpeedSteer.velocity < 2
        return None, other_inputs, stands_inputs
    
    #Override
    def makeNetUsableOtherInputs(self, other_inputs): #normally, the otherinputs are stored as compact as possible. Networks may need to unpack that.
        return other_inputs


    def runInference(self, gameState, pastState): #since we don't have a memory in this agent, we don't care for other_inputs_toSave
        if self.isinitialized and self.checkIfInference():
            conv_inputs, other_inputs, stands_inputs = self.getAgentState(*gameState)
            super().preRunInference()
                
            self.lock.acquire()
            try:
                self.isbusy = True 
                toUse, toSave = self.performNetwork(conv_inputs, self.makeNetUsableOtherInputs(other_inputs), stands_inputs)
                super().postRunInference(toUse, toSave)
                self.isbusy = False
            finally:
                self.lock.release()
                
                

    def performNetwork(self, conv_inputs, inflated_other_inputs, stands_inputs):
        super().performNetwork(conv_inputs, inflated_other_inputs, stands_inputs)
        networkresult, _ = self.cnn.run_inference(self.session, conv_inputs, inflated_other_inputs, stands_inputs) 
        throttle, brake, steer = self.dediscretize(networkresult[0])
        toUse = "["+str(throttle)+", "+str(brake)+", "+str(steer)+"]"
        return toUse, (throttle, brake, steer)

            

    def initNetwork(self):
        initializer = tf.random_uniform_initializer(-0.1, 0.1)
                                     
        with tf.variable_scope("model", reuse=None, initializer=initializer): 
            self.cnn = self.network(self.sv_conf, self, mode="inference")
        
        print(self.cnn.trainvars)
        
        self.saver = tf.train.Saver(self.cnn.trainvars)
        self.session = tf.Session()
        ckpt = tf.train.get_checkpoint_state(self.folder(self.sv_conf.checkpoint_dir)) 
        assert ckpt and ckpt.model_checkpoint_path, "I need a supervisedly pre-trained net!"
        self.saver.restore(self.session, ckpt.model_checkpoint_path)
        print("network initialized")
        self.isinitialized = True


###############################################################################
if __name__ == '__main__':  
    import config
    conf = config.Config()
    agent = Agent(conf, None)
    agent.svTrain()