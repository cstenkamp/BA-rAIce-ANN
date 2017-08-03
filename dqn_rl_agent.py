# -*- coding: utf-8 -*-
"""
Created on Wed May 10 13:31:54 2017

@author: nivradmin
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework import get_variables
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#====own classes====
from agent import AbstractRLAgent
from myprint import myprint as print
import infoscreen
from efficientmemory import Memory as Efficientmemory

current_milli_time = lambda: int(round(time.time() * 1000))

DONT_COPY_WEIGHTS = [] #["FC1", "FC2"]
DONT_TRAIN = [] #["Conv1", "Conv2", "FC1"]# ["Conv1", "Conv2"]

ONLY_START = False


class Agent(AbstractRLAgent):
    def __init__(self, conf, containers, start_fresh, *args, **kwargs):
        self.name = "dqn_rl_agent" #__file__[__file__.rfind("\\")+1:__file__.rfind(".")]
#        self.memory = Efficientmemory(conf.memorysize, conf, self, conf.history_frame_nr, conf.use_constantbutbigmemory) #dieser agent unterstützt das effiziente memory
        super().__init__(conf, containers, *args, **kwargs)
        self.ff_inputsize = 30
        self.epsilon = self.conf.startepsilon
        self.start_fresh = start_fresh
        self.SAVE_ACTION_AS_ARGMAX = False #legacy und speicher-effizienter ists true, aber dann lässt sich das memory nicht als grundlage für ddpg
        if not start_fresh:
            assert os.path.exists(self.folder(self.conf.pretrain_checkpoint_dir)), "I need a pre-trained model"


    def runInference(self, gameState, pastState):
        if self.isinitialized and self.checkIfInference():
            self.preRunInference(gameState, pastState) #eg. adds to memory
            conv_inputs, other_inputs, _ = self.getAgentState(*gameState)
                
#            ##############DELETETHISPART############## #to check how fast the pure socket connection, whithout ANN, is
#            self.containers.outputval.send_via_senderthread("[1, 0, 0]", self.containers.inputval.CTimestamp, self.containers.inputval.STimestamp)
#            return
#            ##############DELETETHISPART ENDE##############
            
            if self.canLearn() and np.random.random() > self.epsilon:
                toUse, toSave = self.performNetwork(conv_inputs, self.makeNetUsableOtherInputs(other_inputs))
            else:
                toUse, toSave = self.randomAction(gameState[2][0].SpeedSteer.velocity, self.conf)
            
                if len(self.memory) >= self.conf.replaystartsize:
                    try:
                        self.epsilon = min(round(max(self.conf.startepsilon-((self.conf.startepsilon-self.conf.minepsilon)*((self.numIterations-self.conf.replaystartsize)/self.conf.finalepsilonframe)), self.conf.minepsilon), 5), 1)
                    except:
                        self.epsilon = min(round(max(self.epsilon-self.conf.epsilondecrease, self.conf.minepsilon), 5), 1)
                    
                if self.containers.showscreen:
                    infoscreen.print(self.epsilon, containers= self.containers, wname="Epsilon")


            if self.containers.showscreen:
                infoscreen.print(toUse, containers= self.containers, wname="Last command")
                if self.numIterations % 100 == 0:
                    infoscreen.print(self.reinfNetSteps, "Iterations: >"+str(self.numIterations), containers= self.containers, wname="ReinfLearnSteps")

            self.postRunInference(toUse, toSave)
    


    def performNetwork(self, conv_inputs, inflated_other_inputs):        
        super().performNetwork(conv_inputs, inflated_other_inputs)
        onehot, qvals = self.target_model.run_inference(self.session, conv_inputs, inflated_other_inputs) #former is argmax, latter are individual qvals
        throttle, brake, steer = self.dediscretize(onehot[0])
        result = "["+str(throttle)+", "+str(brake)+", "+str(steer)+"]"
        self.showqvals(qvals[0])
        if self.SAVE_ACTION_AS_ARGMAX:
            return result, np.argmax(onehot[0])     #er returned immer toUse, toSave
        else:
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
            

    def create_QLearnInputs_from_MemoryBatch(self, memoryBatch):
        if self.SAVE_ACTION_AS_ARGMAX: 
            oldstates, argmactions, rewards, newstates, resetafters = zip(*memoryBatch)      
            actions = np.zeros([len(argmactions), ((4*self.conf.steering_steps) if self.conf.INCLUDE_ACCPLUSBREAK else (3*self.conf.steering_steps))])
            for i in range(len(argmactions)):
                actions[i][argmactions[i]] = 1
                actualActions = [self.dediscretize(i) for i in actions]
        else:
            oldstates, actualActions, rewards, newstates, resetafters = zip(*memoryBatch)      
            actualActions = [self.memory.make_floats_from_long(i) for i in actualActions]
            actions = [self.discretize(throttle, brake, steer) for throttle, brake, steer in actualActions]
            argmactions = [np.argmax(i) for i in actions]
        #soooo, actions[x] = [0,0,1,0,0], argmactions[x] = 2, actualActions[x] = (1,0,0)
        
        old_convs = np.array([i[0] for i in oldstates])
        old_other = np.array([self.makeNetUsableOtherInputs(i[1]) for i in oldstates])
        new_convs = np.array([i[0] for i in newstates])
        new_other = np.array([self.makeNetUsableOtherInputs(i[1]) for i in newstates])
        
        oldAgentStates = (old_convs, old_other)
        newAgentStates = (new_convs, new_other)
        return np.array(oldAgentStates), np.array(newAgentStates), np.array(argmactions), np.array(rewards), np.array(resetafters)
        
                
    
    def learnANN(self):   
        batch = self.memory.sample(self.conf.batchsize)
        QLearnInputs = self.create_QLearnInputs_from_MemoryBatch(batch)
        self.q_learn(self.learn_which, *QLearnInputs, self.conf.batchsize)
        
        self.reinfNetSteps += 1
        print("ReinfLearnSteps:", self.reinfNetSteps, level=3)
        if self.containers.showscreen:
            infoscreen.print(self.reinfNetSteps, "Iterations: >"+str(self.numIterations), containers= self.containers, wname="ReinfLearnSteps")
                    
        if self.reinfNetSteps % self.conf.checkpointall == 0 or self.numIterations >= self.conf.train_for:
            self.saveNet()      
            
        if self.learn_which == self.online_model:
            if self.reinfNetSteps % self.conf.copy_target_all == 0:
                self.lock.acquire()
                self.freezeEverything("saveNet")
                with self.graph.as_default():    
                    self.session.run([target.assign(online) for online, target in zip(get_variables(scope="onlinenet"), get_variables(scope="targetnet"))]) #TODO: wenns ewig lnicht funktioniert, das hier mal umdrehen
                    self.session.run(self.target_model.global_step.assign(self.online_model.global_step)) 
                if self.containers.showscreen:
                    infoscreen.print(time.strftime("%H:%M:%S", time.gmtime()), containers= self.containers, wname="Last Targetnet Copy")
                self.unFreezeEverything("saveNet")
                self.lock.release()
        
                self.evaluator.add_targetnetcopy(ReinfNetSteps=self.reinfNetSteps, MemNum=self.memory._pointer, Iterations=self.numIterations, Episode=self.episodes)



                
                
                
                
    def saveNet(self):
#        self.freezeEverything("saveNet") #TODO: diese auskommentierungen wieder weg!
        checkpoint_file = os.path.join(self.folder(self.conf.checkpoint_dir), 'model.ckpt')
        self.saver.save(self.session, checkpoint_file, global_step=self.online_model.global_step) #remember that this saver only handles the online-net  
#        if self.conf.save_memory_with_checkpoint:
#            self.memory.save_memory()
        print("saved", level=6)
#        self.unFreezeEverything("saveNet")
                
                
    #calculateReward ist in der AbstractRLAgent von der er erbt

        
            
    #randomAction ist ebenfalls in der AbstractRLAgent     
            

    def initNetwork(self): 
        super().initNetwork()
         
        self.session = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=2, allow_soft_placement=True))
        ckpt = tf.train.get_checkpoint_state(self.folder(self.conf.checkpoint_dir))
        initializer = tf.random_uniform_initializer(-0.1, 0)
        self.numIterations = 0
        
        if self.start_fresh:
            print("Initializing the network from scratch", level=9)
            with tf.variable_scope("targetnet", reuse=None, initializer=initializer):
                self.target_model = self.usesnetwork(self.conf, self)                
            with tf.variable_scope("onlinenet", reuse=None, initializer=initializer):
                self.online_model = self.usesnetwork(self.conf, self)                
            init = tf.global_variables_initializer()
            self.session.run(init)        
            self.saver = tf.train.Saver(max_to_keep=1, var_list=get_variables("onlinenet"))
            
            self.session.run([target.assign(online) for online, target in zip(get_variables(scope="onlinenet"), get_variables(scope="targetnet"))])
            
#           
            
        else:
            if ckpt and ckpt.model_checkpoint_path:
             
                print("Initializing the network from the last RL-Save", level=9)
                with tf.variable_scope("targetnet", reuse=None, initializer=initializer):
                    self.target_model = self.usesnetwork(self.conf, self)
                with tf.variable_scope("onlinenet", reuse=None, initializer=initializer):
                    self.online_model = self.usesnetwork(self.conf, self)                                            
                self.saver = tf.train.Saver(max_to_keep=1, var_list=get_variables("onlinenet"))
                self.saver.restore(self.session, ckpt.model_checkpoint_path)
                self.session.run([target.assign(online) for online, target in zip(get_variables(scope="onlinenet"), get_variables(scope="targetnet"))])
                self.reinfNetSteps = self.online_model.global_step.eval(session=self.session)
                self.numIterations = self.online_model.restoreNumIters(self.session)
        
        print("network initialized with %i reinfNetSteps and %i Iterations already run." % (self.reinfNetSteps, self.numIterations))
        self.isinitialized = True
        self.learn_which = self.online_model  
            
            
            

    #requires as input the AGENT-State and the AGENT-paststate: old_convs, old_other, new_convs, new_other
    def q_learn(self, network, old_inputs, new_inputs, argmactions, rewards, resetafters, batchsize):
        
        old_convs, old_other = old_inputs
        new_convs, new_other = new_inputs
        
        qs, max_qs = network.rl_learn_forward(self.session, old_convs, old_other, new_convs, new_other)
        
        consider_stateval = list(np.ones_like(resetafters)-np.array(resetafters, dtype=int))
        
        # wenn folgende Zeile da ist klappt es einigermassen, sonst nicht
        qs = np.zeros_like(qs)
        
        qs[np.arange(batchsize), argmactions] = rewards + self.conf.q_decay * max_qs * consider_stateval #wenn anschließend resettet wurde war es bspw ein wallhit und damit quasi ein final state
          
        network.rl_learn_step(self.session, old_convs, old_other, qs)
           
            

def eraseneccessary(fromwhat, erasewhat):
    topop = []
    for key, _ in fromwhat.items():
        for curr in erasewhat:
            if curr in key:
                topop.append(key)
    for i in topop:
        fromwhat.pop(i)
    return fromwhat  


def print_trainables(session):
    variables_names = [v.name for v in tf.trainable_variables()]
    values = session.run(variables_names)
    for k, v in zip(variables_names, values):
        print("Variable: ", k)
        print("Shape: ", v.shape)
        #print(v)


###############################################################################
if __name__ == '__main__':  
    import config
    conf = config.config()
    from server import Containers; containers = Containers(); containers.conf = conf; containers.conf = conf
    
    myAgent = Agent(conf, containers, True)
#    myAgent.svTrain()
    

    myAgent.initNetwork() #created online_model emptily, dank start_fresh
    #jetzt ne inference q-value anschauen, dann versuchen nen bisschen off-policy rl zu lernen, dann wieder inference anschauen
    
    import read_supervised
    trackingpoints = read_supervised.TPList(conf.LapFolderName, conf.use_second_camera, conf.msperframe, conf.steering_steps, conf.INCLUDE_ACCPLUSBREAK)
    
    #evaluating it BEFORE
    trackingpoints.reset_batch()
    stateBatch, _ = trackingpoints.next_batch(conf, myAgent, trackingpoints.numsamples)
    ev, _, _ = myAgent.online_model.run_sv_eval(myAgent.session, myAgent, stateBatch)                       
    print("Correct inferences: %.2f%%" % (ev*100), level=10)                      

    #doing stuff
    print("Number of samples:",trackingpoints.numsamples)
    for i in range(200):
        trackingpoints.reset_batch()
        while trackingpoints.has_next(32):
            QLearnInputs = read_supervised.create_QLearnInputs_from_SVStateBatch(*trackingpoints.next_batch(conf, myAgent, 32), myAgent)
            myAgent.q_learn(myAgent.online_model, *QLearnInputs, 32)
    
        if (i+1) % 10 == 0:
            myAgent.saveNet()
            
        #evaluating it AFTER
        trackingpoints.reset_batch()
        stateBatch, _ = trackingpoints.next_batch(conf, myAgent, trackingpoints.numsamples)
        ev, _, _ = myAgent.online_model.run_sv_eval(myAgent.session, myAgent, stateBatch)                       
        print("Iteration: %i    Correct inferences: %.2f%%" % (i, ev*100), level=10)                  



    trackingpoints.reset_batch()
    for i in range(10):
        (conv_inputs, other_inputs), _, ArgmActions, _, _ = read_supervised.create_QLearnInputs_from_SVStateBatch(*trackingpoints.next_batch(conf, myAgent, 1), myAgent)
        
        conv_inputs = np.squeeze(np.array(conv_inputs))
        other_inputs = other_inputs[0]
        
        oh, q = myAgent.online_model.run_inference(myAgent.session, conv_inputs, other_inputs)
        
        print(np.argmax(np.array(oh[0])), "  ", ArgmActions[0])
    
        print(q)
    
    
    
    time.sleep(999)
    
    
    
    
    