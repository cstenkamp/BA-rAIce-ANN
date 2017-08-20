# -*- coding: utf-8 -*-
"""
Created on Wed May 10 13:33:46 2017

@author: nivradmin
"""

import tensorflow as tf
import time
#====own classes====
from agent import AbstractAgent
from myprint import myprint as print
from dddqn import DDDQN_model 



class Agent(AbstractAgent):
    def __init__(self, conf, containers, isPretrain=False, start_fresh=False, *args, **kwargs): #der dummy ist da damit man playnet & reinfnet austauschen kan
        self.name = __file__[__file__.rfind("\\")+1:__file__.rfind(".")]
        super().__init__(conf, containers, *args, **kwargs)
        self.ff_inputsize = 30
        self.isSupervised = True
        session = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=2, allow_soft_placement=True))
        self.model = DDDQN_model(self.conf, self, session, isPretrain=isPretrain)
        self.model.initNet(load=(False if start_fresh else "preTrain"))

        
    def runInference(self, gameState, pastState): #since we don't have a memory in this agent, we don't care for other_inputs_toSave
        if self.isinitialized and self.checkIfInference():
            conv_inputs, other_inputs, stands_inputs = self.getAgentState(*gameState)
            toUse, toSave = self.performNetwork(self.makeInferenceUsable((conv_inputs, other_inputs, stands_inputs)))
            self.postRunInference(toUse, toSave)
                

    def performNetwork(self, state):        
        super().performNetwork(state)
        action, qvals = self.model.inference(state) #former is argmax, latter are individual qvals
        throttle, brake, steer = self.dediscretize(action[0])
        result = "["+str(throttle)+", "+str(brake)+", "+str(steer)+"]"
        return result, (throttle, brake, steer) #er returned immer toUse, toSave
            

    def initForDriving(self, *args, **kwargs): 
        super().initForDriving()
        self.isinitialized = True
            
        
    def preTrain(self, dataset, iterations):
        print("Starting pretraining", level=10)
        pretrain_batchsize = self.conf.pretrain_batch_size
        for i in range(iterations):
            start_time = time.time()
            dataset.reset_batch()
            trainBatch = read_supervised.create_QLearnInputs_from_PTStateBatch(*dataset.next_batch(self.conf, self, dataset.numsamples), self)
            print('Iteration %3d: Accuracy = %.2f%% (%.1f sec)' % (self.model.pretrain_episode(), self.model.getAccuracy(trainBatch), time.time()-start_time), level=10)
            self.model.inc_episode()
            dataset.reset_batch()
            while dataset.has_next(pretrain_batchsize):
                trainBatch = read_supervised.create_QLearnInputs_from_PTStateBatch(*dataset.next_batch(self.conf, self, pretrain_batchsize), self)
                self.model.sv_learn(trainBatch, True)
            if (i+1) % 25 == 0:
                self.model.save()    
        
        
        
###############################################################################
if __name__ == '__main__':  
    import sys
    import config
    conf = config.Config()
    import read_supervised
    from server import Containers; containers = Containers()
    tf.reset_default_graph()                                                          
    myAgent = Agent(conf, containers, isPretrain = True, start_fresh=("-new" in sys.argv))
    trackingpoints = read_supervised.TPList(conf.LapFolderName, conf.use_second_camera, conf.msperframe, conf.steering_steps, conf.INCLUDE_ACCPLUSBREAK)
    print("Number of samples:",trackingpoints.numsamples)
    myAgent.preTrain(trackingpoints, 200)
    time.sleep(999)