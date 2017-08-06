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
    def __init__(self, conf, containers, start_fresh, *args, **kwargs):
        self.name = "dqn_rl_agent" #__file__[__file__.rfind("\\")+1:__file__.rfind(".")]
#        self.memory = Efficientmemory(conf.memorysize, conf, self, conf.history_frame_nr, conf.use_constantbutbigmemory) #dieser agent unterstützt das effiziente memory
        super().__init__(conf, containers, *args, **kwargs)
        self.ff_inputsize = 30
        self.epsilon = self.conf.startepsilon
        self.start_fresh = start_fresh
        if not start_fresh:
            assert os.path.exists(self.folder(self.conf.pretrain_checkpoint_dir) or self.folder(self.conf.checkpoint_dir)), "I need any kind of pre-trained model"


    def runInference(self, gameState, pastState):
        if self.isinitialized and self.checkIfInference():
            self.preRunInference(gameState, pastState) #eg. adds to memory
            conv_inputs, other_inputs, stands_inputs = self.getAgentState(*gameState)
                            
#            ##############DELETETHISPART############## #to check how fast the pure socket connection, whithout ANN, is
#            self.containers.outputval.send_via_senderthread("[1, 0, 0]", self.containers.inputval.CTimestamp, self.containers.inputval.STimestamp)
#            return
#            ##############DELETETHISPART ENDE##############
            
            if self.canLearn() and np.random.random() > self.epsilon:
                toUse, toSave = self.performNetwork(self.makeInferenceUsable((conv_inputs, other_inputs, stands_inputs)))
            else:
                toUse, toSave = self.randomAction(gameState[2][0].SpeedSteer.velocity)
            
                if len(self.memory) >= self.conf.replaystartsize:
                    try:
                        self.epsilon = min(round(max(self.conf.startepsilon-((self.conf.startepsilon-self.conf.minepsilon)*((self.model.run_inferences()-self.conf.replaystartsize)/self.conf.finalepsilonframe)), self.conf.minepsilon), 5), 1)
                    except: #there are two different kinds of what can be stored in the config for the memory-decrease
                        self.epsilon = min(round(max(self.epsilon-self.conf.epsilondecrease, self.conf.minepsilon), 5), 1)
                    
                if self.containers.showscreen:
                    infoscreen.print(self.epsilon, containers=self.containers, wname="Epsilon")

            if self.containers.showscreen:
                infoscreen.print(toUse, containers=self.containers, wname="Last command")
                if self.model.run_inferences() % 100 == 0:
                    infoscreen.print(self.model.step(), "Iterations: >"+str(self.model.run_inferences()), containers=self.containers, wname="ReinfLearnSteps")

            self.postRunInference(toUse, toSave)
    


    def performNetwork(self, state):        
        super().performNetwork(state)
        action, qvals = self.model.inference(state) #former is argmax, latter are individual qvals
        throttle, brake, steer = self.dediscretize(action[0])
        result = "["+str(throttle)+", "+str(brake)+", "+str(steer)+"]"
        self.showqvals(qvals[0])
        return result, (throttle, brake, steer) #er returned immer toUse, toSave




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


    #dauerlearnANN kommt aus der AbstractRLAgent
        
    
    #memoryBatch is [[s,a,r,s2,t],[s,a,r,s2,t],[s,a,r,s2,t],...], what we want as Q-Learn-Input is [[s],[a],[r],[s2],[t]] 
    #.. to be more precise: [[(c,f),a,r,(c,f),t],[(c,f),a,r,(c,f),t],...]  and [[(c,f)],[a],[r],[(c,f)],[t]]
    def create_QLearnInputs_from_MemoryBatch(self, memoryBatch):
        oldstates, actions, rewards, newstates, resetafters = zip(*memoryBatch)      
        #is already [[(c,f)],[a],[r],[(c,f)],[t]], however the actions are tuples, and we want argmax's... and netUsableOtherinputs
        actions = np.array([np.argmax(self.makeNetUsableAction((throttle, brake, steer))) for throttle, brake, steer in actions]) 
        oldstates = [(np.rollaxis(np.array(i[0]), 0, 3), np.array(self.makeNetUsableOtherInputs(i[1]))) for i in oldstates]
        newstates = [(np.rollaxis(np.array(i[0]), 0, 3), np.array(self.makeNetUsableOtherInputs(i[1]))) for i in newstates]#
        return oldstates, actions, np.array(rewards), newstates, np.array(resetafters)
        
                
    
    def learnANN(self):   
        QLearnInputs = self.create_QLearnInputs_from_MemoryBatch(self.memory.sample(self.conf.batch_size))
        self.model.q_learn(QLearnInputs, False)
        
        print("ReinfLearnSteps:", self.model.step(), level=3)
        if self.containers.showscreen:
            infoscreen.print(self.model.step(), "Iterations: >"+str(self.model.run_inferences()), containers= self.containers, wname="ReinfLearnSteps")
                    
        if self.model.step() > 0 and self.model.step() % self.conf.checkpointall == 0 or self.model.run_inferences() >= self.conf.train_for:
            self.saveNet()      
                            
                
    #calculateReward ist in der AbstractRLAgent von der er erbt

            
    #randomAction ist ebenfalls in der AbstractRLAgent     
            

    def initNetwork(self, isPretrain): 
        super().initNetwork()
        session = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=2, allow_soft_placement=True))
        self.model = DDDQN_model(self.conf, self, session, isPretrain=isPretrain)
        self.model.initNet(load=(not self.start_fresh))
        self.isinitialized = True
            
        
    
    def preTrain(self, dataset, iterations, supervised=False):
        print("Starting pretraining", level=10)
        pretrain_batchsize = 32
        for i in range(iterations):
            start_time = time.time()
            dataset.reset_batch()
            trainBatch = read_supervised.create_QLearnInputs_from_PTStateBatch(*dataset.next_batch(self.conf, self, dataset.numsamples), self)
            print('Iteration %3d: Accuracy = %.2f%% (%.1f sec)' % (self.model.pretrain_episode(), self.model.getAccuracy(trainBatch), time.time()-start_time), level=10)
            self.model.inc_episode()
            dataset.reset_batch()
            while dataset.has_next(pretrain_batchsize):
                trainBatch = read_supervised.create_QLearnInputs_from_PTStateBatch(*dataset.next_batch(self.conf, self, pretrain_batchsize), self)
                if supervised:
                    self.model.sv_learn(trainBatch, True)
                else:
                    self.model.q_learn(trainBatch, True)    
                    
            if (i+1) % 25 == 0:
                self.saveNet()
        
            
###############################################################################

if __name__ == '__main__':  
    import config
    conf = config.Config()
    import read_supervised
    from server import Containers; containers = Containers()
    tf.reset_default_graph()                                                          
    myAgent = Agent(conf, containers, start_fresh=True)
    myAgent.initNetwork(isPretrain=True)
    trackingpoints = read_supervised.TPList(conf.LapFolderName, conf.use_second_camera, conf.msperframe, conf.steering_steps, conf.INCLUDE_ACCPLUSBREAK)
    print("Number of samples:",trackingpoints.numsamples)
    myAgent.preTrain(trackingpoints, 200)
    time.sleep(999)