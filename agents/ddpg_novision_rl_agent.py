# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 15:47:57 2017

@author: nivradmin
"""


import numpy as np
import tensorflow as tf
import time
#====own classes====
from agent import AbstractRLAgent
from myprint import myprint as print
from efficientmemory import Memory as Efficientmemory
from ddpg import DDPG_model 
import infoscreen

flatten = lambda l: [item for sublist in l for item in sublist]


class Agent(AbstractRLAgent):
    def __init__(self, conf, containers, isPretrain=False, start_fresh=False, *args, **kwargs):
        self.name = "ddpg_novision_rl_agent" #__file__[__file__.rfind("\\")+1:__file__.rfind(".")]
        super().__init__(conf, containers, isPretrain, start_fresh, *args, **kwargs)
        self.ff_inputsize = 49 + conf.num_actions * conf.ff_stacksize #61
        self.isContinuous = True
        self.usesConv = False
        self._noiseState = np.array([0]*self.conf.num_actions)
        session = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=2, allow_soft_placement=True))
        self.model = DDPG_model(self.conf, self, session, isPretrain=isPretrain)
        self.model.initNet(load=("preTrain" if (self.isPretrain and not start_fresh) else (not start_fresh)))


    ###########################################################################
    ########################overwritten functions##############################
    ###########################################################################
    
    #im gegensatz zu den DQN-basierten agents muss er die action nicht diskretisieren
    def makeNetUsableAction(self, action):
        return action

    def getAgentState(self, *gameState):  
        vvec1_hist, vvec2_hist, otherinput_hist, action_hist = gameState
        flat_actions = flatten([i if i is not None else (0,0,0) for i in action_hist])
#        other_inputs = np.ravel([i.returnRelevant() for i in otherinput_hist])
        other_inputs = np.ravel(otherinput_hist[0].returnRelevant()); other_inputs = np.concatenate((other_inputs,flat_actions))
        stands_inputs = otherinput_hist[0].SpeedSteer.velocity < 10
        return None, other_inputs, stands_inputs
    
    def makeNetUsableOtherInputs(self, other_inputs): #normally, the otherinputs are stored as compact as possible. Networks may need to unpack that.
        return other_inputs
    
    def endEpisode(self, *args, **kwargs):
        self._noiseState = np.array([0]*self.conf.num_actions)
        super().endEpisode(*args, **kwargs)


    ###########################################################################
    ########################functions that need to be implemented##############
    ###########################################################################
    
    def initForDriving(self, *args, **kwargs): 
#        self.memory = Efficientmemory(self.conf.memorysize, self.conf, self, self.conf.history_frame_nr, self.conf.use_constantbutbigmemory) #dieser agent unterstützt das effiziente memory        
        super().initForDriving(*args, **kwargs)
        self.isinitialized = True


    #classical Ornstein-Uhlenbeck-process. The trick in that is, that the mu of the noise is always that one of the last iteration (->temporal correlation)
    def make_noisy(self, action):
        self._noiseState = self.conf.ornstein_theta * self._noiseState + (1-self.conf.ornstein_theta) * np.random.normal(np.zeros_like(self._noiseState), self.conf.ornstein_std)
        action = action + 10*self.epsilon * self._noiseState
        clip = lambda x,b: min(max(x,b[0]),b[1])
        action = np.array([clip(action[i],self.conf.action_bounds[i]) for i in range(len(action))])
        return action


    #because we don't do epsilon-greedy here but ONP, we let randomAction be policyAction... but only once we filled the memory enough
    def randomAction(self, agentState):
        if len(self.memory) > self.conf.replaystartsize:
            return self.policyAction(agentState)
        else:
            return super().randomAction(agentState)
    
    
    def policyAction(self, agentState):
        action, _ = self.model.inference(self.makeInferenceUsable(agentState))
        action = self.make_noisy(action[0])
        action = [round(i,3) for i in action]
        toUse = "["+str(action[0])+", "+str(action[1])+", "+str(action[2])+"]"
        if self.containers.showscreen:
            infoscreen.print(toUse, containers=self.containers, wname="Last command")
            if self.model.run_inferences() % 100 == 0:
                infoscreen.print(self.model.step(), "Iterations: >"+str(self.model.run_inferences()), containers=self.containers, wname="ReinfLearnSteps")
                infoscreen.print(self.epsilon, containers=self.containers, wname="Epsilon")
        return toUse, action



    def make_trainbatch(self,dataset,batchsize,epsilon=0):
        clip = lambda x,b: min(max(x,b[0]),b[1])
        trainBatch = dataset.create_QLearnInputs_fromBatch(*dataset.next_batch(self.conf, self, batchsize), self)
        if epsilon > np.random.random():
            s,a,r,s2,t = trainBatch
            a2 = a + np.random.normal(np.zeros_like(a), epsilon*self.conf.ornstein_std)
            a2 = np.array([[clip(curr_a[i],self.conf.action_bounds[i]) for i in range(len(curr_a))] for curr_a in a2])
            rewarddiff = [1-min(np.linalg.norm(a2[i]-a[i]),1) for i in range(len(a))]
            r = [r[i]*rewarddiff[i] if r[i] > 0 else r[i]*(1+rewarddiff[i]) for i in range(len(rewarddiff))]
            trainBatch = s,a2,r,s2,t
        return trainBatch


    def preTrain(self, dataset, iterations, supervised=False):
        assert self.model.step() == 0, "I dont pretrain if the model already learned on real data!"
        iterations = self.conf.pretrain_iterations if iterations is None else iterations
        if supervised:
            raise ValueError("A DDPG-Model cannot learn supervisedly!")
        print("Starting pretraining", level=10)
        for i in range(iterations):
            start_time = time.time()
            self.model.inc_episode()
            dataset.reset_batch()
            while dataset.has_next(self.conf.pretrain_batch_size):
                trainBatch = self.make_trainbatch(dataset,self.conf.pretrain_batch_size,0.8)
                self.model.q_train_step(trainBatch, True)    
            if (i+1) % 25 == 0:
                self.model.save()    
            dataset.reset_batch()
            trainBatch = self.make_trainbatch(dataset,dataset.numsamples)
            print('Iteration %3d: Closeness = %.2f (%.1f sec)' % (self.model.pretrain_episode(), self.model.getAccuracy(trainBatch), time.time()-start_time), level=10)
            


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
            
            