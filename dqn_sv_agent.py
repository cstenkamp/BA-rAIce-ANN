# -*- coding: utf-8 -*-
"""
Created on Wed May 10 13:33:46 2017

@author: nivradmin
"""

import tensorflow as tf

#====own classes====
from agent import AbstractAgent
from myprint import myprint as print



class Agent(AbstractAgent):
    def __init__(self, conf, containers, startfresh_dummy=False, *args, **kwargs): #der dummy ist da damit man playnet & reinfnet austauschen kan
        self.name = "dqn_rl_agent" #weil das der selbe ist, nur halt dass hier das lernen nicht rein-implementiert ist
        super().__init__(conf, containers, *args, **kwargs)
        self.ff_inputsize = 30


    def runInference(self, gameState, pastState): #since we don't have a memory in this agent, we don't care for other_inputs_toSave
        if self.isinitialized and self.checkIfInference():
            conv_inputs, other_inputs, stands_inputs = self.getAgentState(*gameState)
            super().preRunInference()
                
            self.lock.acquire()
            try:
                self.isbusy = True 
                toUse, toSave = self.performNetwork(conv_inputs, self.makeNetUsableOtherInputs(other_inputs), stands_inputs) #This now also needs makeInferenceUsable!!!
                super().postRunInference(toUse, toSave) 
                self.isbusy = False
            finally:
                self.lock.release()
                  
                

    def performNetwork(self, conv_inputs, inflated_other_inputs, stands_inputs):
        super().performNetwork(conv_inputs, inflated_other_inputs, stands_inputs)
        networkresult, _ = self.model.run_inference(self.session, conv_inputs, inflated_other_inputs, stands_inputs) 
        throttle, brake, steer = self.dediscretize(networkresult[0])
        toUse = "["+str(throttle)+", "+str(brake)+", "+str(steer)+"]"
        return toUse, (throttle, brake, steer)

            

    def initNetwork(self):        
        initializer = tf.random_uniform_initializer(-0.1, 0.1)
                                           
        with tf.variable_scope("model", reuse=None, initializer=initializer): 
            self.model = self.usesnetwork(self.conf, self, mode="inference")
        
        print(self.model.trainvars)
        
        self.saver = tf.train.Saver(self.model.trainvars)
        self.session = tf.Session()
        ckpt = tf.train.get_checkpoint_state(self.folder(self.conf.checkpoint_dir))
        assert ckpt and ckpt.model_checkpoint_path, "I need a supervisedly pre-trained net!"
        self.saver.restore(self.session, ckpt.model_checkpoint_path)
        print("network initialized")
        self.isinitialized = True
            
            
###############################################################################
if __name__ == '__main__':  
#    import config
#    conf = config.Config()
#    agent = Agent(conf, None)
#    agent.svTrain()