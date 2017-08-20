# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 15:47:57 2017

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
from efficientmemory import Memory as Efficientmemory
from ddpg import DDPG_model 




class Agent(AbstractRLAgent):
    def __init__(self, conf, containers, isPretrain=False, start_fresh=False, *args, **kwargs):
        self.name = "dqn_rl_agent" #__file__[__file__.rfind("\\")+1:__file__.rfind(".")]
        super().__init__(conf, containers, isPretrain, start_fresh, *args, **kwargs)
        self.ff_inputsize = 30
        self.isContinuous = True
        self.usesConv = False
        self._noiseState = [0]*self.conf.num_actions
        session = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=2, allow_soft_placement=True))
        self.model = DDPG_model(self.conf, self, session, isPretrain=isPretrain)
        self.model.initNet(load=("preTrain" if (self.isPretrain and not start_fresh) else (not start_fresh)))



    ###########################################################################
    ########################overwritten functions##############################
    ###########################################################################
    
    #im gegensatz zu den DQN-basierten agents muss er die action nicht diskretisieren
    #Override
    def makeNetUsableAction(self, action):
        return action

    #Override
    def getAgentState(self, *gameState):  
        vvec1_hist, vvec2_hist, otherinput_hist, action_hist = gameState
#        other_inputs = np.ravel([i.returnRelevant() for i in otherinput_hist])
        other_inputs = np.ravel(otherinput_hist[0].returnRelevant())
        stands_inputs = otherinput_hist[0].SpeedSteer.velocity < 10
        return None, other_inputs, stands_inputs
    
    #Override
    def makeNetUsableOtherInputs(self, other_inputs): #normally, the otherinputs are stored as compact as possible. Networks may need to unpack that.
        return other_inputs

            

    ###########################################################################
    ########################functions that need to be implemented##############
    ###########################################################################
    
    def initForDriving(self, *args, **kwargs): 
#        self.memory = Efficientmemory(self.conf.memorysize, self.conf, self, self.conf.history_frame_nr, self.conf.use_constantbutbigmemory) #dieser agent unterstützt das effiziente memory        
        super().initForDriving(*args, **kwargs)
        self.isinitialized = True


    #classical Ornstein-Uhlenbeck-process. The trick in that is, that the mu of the noise is always that one of the last iteration (->temporal correlation)
    def make_noisy(self, action):
        self._noiseState = self.conf.ornstein_theta * self._noiseState + np.random.normal(np.zeros_like(self._noiseState), self.conf.ornstein_std)
        action = action + self.epsilon * self._noiseState
        clip = lambda x,b: min(max(x,b[0]),b[1])
        action = [clip(action[i],self.conf.action_bounds[i]) for i in range(len(action))]
        return action


    #because we don't do epsilon-greedy here but ONP, we let randomAction be policyAction... but only once we filled the memory enough
    def randomAction(self, agentState):
        if len(self.memory) > self.conf.replaystartsize:
            return self.policyAction(agentState)
        else:
            return super().randomAction(agentState)
    
    
    def policyAction(self, agentState):
        action, _ = self.model.inference(self.makeInferenceUsable(agentState))
        action = self.make_noisy(action)
        toUse = "["+str(action[0])+", "+str(action[1])+", "+str(action[2])+"]"
        return toUse, action


    def preTrain(self, dataset, Iterations, supervised=False):
        if supervised:
            raise ValueError("A DDPG-Model cannot learn supervisedly!")
        raise NotImplementedError

#    def preTrain(self, dataset, iterations, supervised=False):
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


        
    
            
###############################################################################

#if __name__ == '__main__':  
#    import config
#    conf = config.Config()
#    import read_supervised
#    from server import Containers; containers = Containers()
#    tf.reset_default_graph()                                                          
#    myAgent = Agent(conf, containers, start_fresh=True, isPretrain=True)
#    trackingpoints = read_supervised.TPList(conf.LapFolderName, conf.use_second_camera, conf.msperframe, conf.steering_steps, conf.INCLUDE_ACCPLUSBREAK)
#    print("Number of samples:",trackingpoints.numsamples)
#    myAgent.preTrain(trackingpoints, 200)
#    time.sleep(999)