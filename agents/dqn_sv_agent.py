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
        self.ff_inputsize = conf.speed_neurons + conf.num_actions * conf.ff_stacksize #32
        self.isSupervised = True
        session = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=2, allow_soft_placement=True))
        self.model = DDDQN_model(self.conf, self, session, isPretrain=isPretrain)
        self.model.initNet(load=(False if start_fresh else "preTrain"))

        
    def policyAction(self, agentState):
        action, qvals = self.model.inference(self.makeInferenceUsable(agentState)) #former is argmax, latter are individual qvals
        print(action)
        throttle, brake, steer = self.dediscretize(action[0])
        toUse = "["+str(throttle)+", "+str(brake)+", "+str(steer)+"]"
        return toUse, (throttle, brake, steer)

            
        
    def preTrain(self, dataset, iterations, supervised=True):
        assert supervised, "the SV-agent can only train supervisedly!"
        print("Starting pretraining", level=10)
        pretrain_batchsize = self.conf.pretrain_batch_size
        for i in range(iterations):
            start_time = time.time()
            self.model.inc_episode()
            dataset.reset_batch()
            while dataset.has_next(pretrain_batchsize):
                trainBatch = dataset.create_QLearnInputs_fromBatch(*dataset.next_batch(self.conf, self, pretrain_batchsize), self)
                self.model.sv_train_step(trainBatch, True)
            if (i+1) % 25 == 0:
                self.model.save()    
            dataset.reset_batch()
            trainBatch = dataset.create_QLearnInputs_fromBatch(*dataset.next_batch(self.conf, self, dataset.numsamples), self)
            print('Iteration %3d: Accuracy = %.2f%% (%.1f sec)' % (self.model.pretrain_episode(), self.model.getAccuracy(trainBatch, likeDDPG=False), time.time()-start_time), level=10)
        
