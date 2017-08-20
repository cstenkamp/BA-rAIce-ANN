# -*- coding: utf-8 -*-
"""
Created on Wed May 10 13:31:54 2017

@author: nivradmin
"""

import numpy as np
import tensorflow as tf
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#====own classes====
from agent import AbstractRLAgent
from myprint import myprint as print
import infoscreen
from efficientmemory import Memory as Efficientmemory
from dddqn import DDDQN_model 

current_milli_time = lambda: int(round(time.time() * 1000))



class Agent(AbstractRLAgent):    
    def __init__(self, conf, containers, isPretrain=False, start_fresh=False, *args, **kwargs):
        self.name = "dqn_rl_agent" #__file__[__file__.rfind("\\")+1:__file__.rfind(".")]
        super().__init__(conf, containers, isPretrain, start_fresh, *args, **kwargs)
        self.ff_inputsize = 30
        session = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=2, allow_soft_placement=True))
        self.model = DDDQN_model(self.conf, self, session, isPretrain=isPretrain)
        self.model.initNet(load=("preTrain" if (self.isPretrain and not start_fresh) else (not start_fresh)))



    ###########################################################################
    ########################functions that need to be implemented##############
    ###########################################################################
    
    def initForDriving(self, *args, **kwargs): 
#        self.memory = Efficientmemory(self.conf.memorysize, self.conf, self, self.conf.history_frame_nr, self.conf.use_constantbutbigmemory) #dieser agent unterstützt das effiziente memory        
        super().initForDriving(*args, **kwargs)
        self.isinitialized = True



    def policyAction(self, agentState):
        action, qvals = self.model.inference(self.makeInferenceUsable(agentState)) #former is argmax, latter are individual qvals
        throttle, brake, steer = self.dediscretize(action[0])
        toUse = "["+str(throttle)+", "+str(brake)+", "+str(steer)+"]"
        self.showqvals(qvals[0])
        if self.containers.showscreen:
            infoscreen.print(toUse, containers=self.containers, wname="Last command")
        if self.containers.showscreen:
            if self.model.run_inferences() % 100 == 0:
                infoscreen.print(self.model.step(), "Iterations: >"+str(self.model.run_inferences()), containers=self.containers, wname="ReinfLearnSteps")
        return toUse, (throttle, brake, steer) #er returned immer toUse, toSave

    

    def randomAction(self, agentState):
        toUse, toSave = super().randomAction(agentState)
        if self.containers.showscreen:
            infoscreen.print(toUse, "(random)", containers=self.containers, wname="Last command")
            infoscreen.print(self.epsilon, containers=self.containers, wname="Epsilon")
        return toUse, toSave


#    def preTrain(self, dataset, iterations, supervised=False):
##        assert self.model.step == 0, "I dont pretrain if the model already learned on real data!"
#        print("Starting pretraining", level=10)
#        pretrain_batchsize = 32
#        for i in range(iterations):
#            start_time = time.time()
#            dataset.reset_batch()
#            trainBatch = read_supervised.create_QLearnInputs_from_PTStateBatch(*dataset.next_batch(self.conf, self, dataset.numsamples), self)
#            print('Iteration %3d: Accuracy = %.2f%% (%.1f sec)' % (self.model.pretrain_episode(), self.model.getAccuracy(trainBatch), time.time()-start_time), level=10)
#            self.model.inc_episode()
#            dataset.reset_batch()
#            while dataset.has_next(pretrain_batchsize):
#                trainBatch = read_supervised.create_QLearnInputs_from_PTStateBatch(*dataset.next_batch(self.conf, self, pretrain_batchsize), self)
#                if supervised:
#                    self.model.sv_learn(trainBatch, True)
#                else:
#                    self.model.q_learn(trainBatch, True)    
#                    
#            if (i+1) % 25 == 0:
#                self.saveNet()

    ###########################################################################
    ########################overwritten functions##############################
    ###########################################################################
    
    def eval_episodeVals(self, mem_epi_slice, gameState, endReason):
        string = super().eval_episodeVals(mem_epi_slice, gameState, endReason)
        if self.containers.showscreen: 
            infoscreen.print(string, containers=self.containers, wname="Last Epsd")

            
    def punishLastAction(self, howmuch):
        super().punishLastAction(howmuch)
        if self.containers.showscreen:
            infoscreen.print(str(-abs(howmuch)), time.strftime("%H:%M:%S", time.gmtime()), containers=self.containers, wname="Last big punish")
            
    def addToMemory(self, gameState, pastState):
        a, r, stateval, changestring = super().addToMemory(gameState, pastState)
        if self.containers.showscreen:
            infoscreen.print(a, round(r,2), round(stateval,2), changestring, containers= self.containers, wname="Last memory")
            if len(self.memory) % 20 == 0:
                infoscreen.print(">"+str(len(self.memory)), containers= self.containers, wname="Memorysize")       
                
    def learnANN(self):  
        super().learnANN()
        print("ReinfLearnSteps:", self.model.step(), level=3)
        if self.containers.showscreen:
            infoscreen.print(self.model.step(), "Iterations: >"+str(self.model.run_inferences()), containers= self.containers, wname="ReinfLearnSteps")                
                
    ###########################################################################
    ########################additional functions###############################
    ###########################################################################
    

    def showqvals(self, qvals):
        amount = self.conf.steering_steps*4 if self.conf.INCLUDE_ACCPLUSBREAK else self.conf.steering_steps*3
        b = []
        for i in range(amount):
            a = [0]*amount
            a[i] = 1
            b.append(str(self.dediscretize(a)))
        b = list(zip(b, qvals))
        toprint = [str(i[0])[1:-1]+": "+str(i[1]) for i in b]
        toprint = "\n".join(toprint)
        print(b, level=3)
        if self.containers.showscreen:
            infoscreen.print(toprint, containers= self.containers, wname="Current Q Vals")

        
