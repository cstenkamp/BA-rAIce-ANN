# -*- coding: utf-8 -*-
"""
Created on Wed May 10 13:31:54 2017

@author: nivradmin
"""

import numpy as np
import tensorflow as tf
import time
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
        self.ff_inputsize = conf.speed_neurons + conf.num_actions * conf.ff_stacksize #32
        session = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=2, allow_soft_placement=True))
        self.usesGUI = True
        self.model = DDDQN_model(self.conf, self, session, isPretrain=isPretrain)
        self.model.initNet(load=("preTrain" if (self.isPretrain and not start_fresh) else (not start_fresh)))



    ###########################################################################
    ########################functions that need to be implemented##############
    ###########################################################################
    
    def initForDriving(self, *args, **kwargs): 
#        self.memory = Efficientmemory(self.conf.memorysize, self.conf, self, self.conf.history_frame_nr, self.conf.use_constantbutbigmemory) #dieser agent unterstützt das effiziente memory        
        super().initForDriving(*args, **kwargs)



    def policyAction(self, agentState):
        action, qvals = self.model.inference(self.makeInferenceUsable(agentState)) #former is argmax, latter are individual qvals
        throttle, brake, steer = self.dediscretize(action[0])
        toUse = "["+str(throttle)+", "+str(brake)+", "+str(steer)+"]"
        self.showqvals(qvals[0])
        if self.containers.showscreen:
            infoscreen.print(toUse, containers=self.containers, wname="Last command")
            if self.model.run_inferences() % 100 == 0:
                infoscreen.print(self.model.step(), "Iterations: >"+str(self.model.run_inferences()), containers=self.containers, wname="ReinfLearnSteps")
        return toUse, (throttle, brake, steer) #er returned immer toUse, toSave

    

    def randomAction(self, agentState):
        toUse, toSave = super().randomAction(agentState)
        if self.containers.showscreen:
            infoscreen.print(toUse, "(random)", containers=self.containers, wname="Last command")
            infoscreen.print(self.epsilon, containers=self.containers, wname="Epsilon")
        return toUse, toSave


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
        a, r, qval, count, changestring = super().addToMemory(gameState, pastState)
        if self.containers.showscreen:
            infoscreen.print(a, round(r,2), round(qval,2), changestring, containers= self.containers, wname="Last memory")
            if len(self.memory) % 20 == 0:
                infoscreen.print(">"+str(len(self.memory)), containers= self.containers, wname="Memorysize")       
                
    def learnANN(self):  
        tmp = super().learnANN()
        print("ReinfLearnSteps:", self.model.step(), level=3)
        if self.containers.showscreen:
            infoscreen.print(self.model.step(), "Iterations: >"+str(self.model.run_inferences()), containers= self.containers, wname="ReinfLearnSteps")                
        return tmp     
            
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

        
