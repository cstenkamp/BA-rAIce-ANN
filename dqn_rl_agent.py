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
    def __init__(self, sv_conf, containers, rl_conf, start_fresh, *args, **kwargs):
        self.name = "dqn_rl_agent" #__file__[__file__.rfind("\\")+1:__file__.rfind(".")]
#        self.memory = Efficientmemory(rl_conf.memorysize, rl_conf, self, rl_conf.history_frame_nr, rl_conf.use_constantbutbigmemory) #dieser agent unterstützt das effiziente memory
        super().__init__(sv_conf, containers, rl_conf, *args, **kwargs)
        self.ff_inputsize = 30
        self.epsilon = self.rl_conf.startepsilon
        self.start_fresh = start_fresh
        self.SAVE_ACTION_AS_ARGMAX = False #legacy und speicher-effizienter ists true, aber dann lässt sich das memory nicht als grundlage für ddpg
        if not start_fresh:
            assert os.path.exists(self.folder(self.sv_conf.checkpoint_dir)), "I need a pre-trained model"


    def runInference(self, gameState, pastState):
        if self.isinitialized and self.checkIfInference():
            self.preRunInference(gameState, pastState) #eg. adds to memory
            conv_inputs, other_inputs, stands_inputs = self.getAgentState(*gameState)
                
#            ##############DELETETHISPART############## #to check how fast the pure socket connection, whithout ANN, is
#            self.containers.outputval.send_via_senderthread("[1, 0, 0]", self.containers.inputval.CTimestamp, self.containers.inputval.STimestamp)
#            return
#            ##############DELETETHISPART ENDE##############
            
            if self.canLearn() and np.random.random() > self.epsilon:
                toUse, toSave = self.performNetwork(conv_inputs, self.makeNetUsableOtherInputs(other_inputs), stands_inputs)
            else:
                toUse, toSave = self.randomAction(gameState[2][0].SpeedSteer.velocity, self.rl_conf)
            
                if len(self.memory) >= self.rl_conf.replaystartsize:
                    try:
                        self.epsilon = min(round(max(self.rl_conf.startepsilon-((self.rl_conf.startepsilon-self.rl_conf.minepsilon)*((self.numIterations-self.rl_conf.replaystartsize)/self.rl_conf.finalepsilonframe)), self.rl_conf.minepsilon), 5), 1)
                    except:
                        self.epsilon = min(round(max(self.epsilon-self.rl_conf.epsilondecrease, self.rl_conf.minepsilon), 5), 1)
                    
                if self.containers.showscreen:
                    infoscreen.print(self.epsilon, containers= self.containers, wname="Epsilon")


            if self.containers.showscreen:
                infoscreen.print(toUse, containers= self.containers, wname="Last command")
                if self.numIterations % 100 == 0:
                    infoscreen.print(self.reinfNetSteps, "Iterations: >"+str(self.numIterations), containers= self.containers, wname="ReinfLearnSteps")

            self.postRunInference(toUse, toSave)
    


    def performNetwork(self, conv_inputs, inflated_other_inputs, stands_inputs):        
        super().performNetwork(conv_inputs, inflated_other_inputs, stands_inputs)
        onehot, qvals = self.target_model.run_inference(self.session, conv_inputs, inflated_other_inputs, stands_inputs) #former is argmax, latter are individual qvals
        throttle, brake, steer = self.dediscretize(onehot[0])
        result = "["+str(throttle)+", "+str(brake)+", "+str(steer)+"]"
        self.showqvals(qvals[0])
        if self.SAVE_ACTION_AS_ARGMAX:
            return result, np.argmax(onehot[0])     #er returned immer toUse, toSave
        else:
            return result, (throttle, brake, steer) #er returned immer toUse, toSave




    def showqvals(self, qvals):
        amount = self.rl_conf.steering_steps*4 if self.rl_conf.INCLUDE_ACCPLUSBREAK else self.rl_conf.steering_steps*3
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
            actions = np.zeros([len(argmactions), ((4*self.rl_conf.steering_steps) if self.rl_conf.INCLUDE_ACCPLUSBREAK else (3*self.rl_conf.steering_steps))])
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
        batch = self.memory.sample(self.rl_conf.batchsize)
        QLearnInputs = self.create_QLearnInputs_from_MemoryBatch(batch)
        self.q_learn(self.learn_which, *QLearnInputs, self.rl_conf.batchsize)
        
        self.reinfNetSteps += 1
        print("ReinfLearnSteps:", self.reinfNetSteps, level=3)
        if self.containers.showscreen:
            infoscreen.print(self.reinfNetSteps, "Iterations: >"+str(self.numIterations), containers= self.containers, wname="ReinfLearnSteps")
                    
        if self.reinfNetSteps % self.rl_conf.checkpointall == 0 or self.numIterations >= self.rl_conf.train_for:
            self.saveNet()      
            
        if self.learn_which == self.online_model:
            if self.reinfNetSteps % self.rl_conf.copy_target_all == 0:
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
        self.online_model.saveNumIters(self.session, self.numIterations)
        checkpoint_file = os.path.join(self.folder(self.rl_conf.checkpoint_dir), 'model.ckpt')
        self.saver.save(self.session, checkpoint_file, global_step=self.online_model.global_step) #remember that this saver only handles the online-net  
#        if self.rl_conf.save_memory_with_checkpoint:
#            self.memory.save_memory()
        print("saved", level=6)
#        self.unFreezeEverything("saveNet")
                
                
    #calculateReward ist in der AbstractRLAgent von der er erbt

        
            
    #randomAction ist ebenfalls in der AbstractRLAgent     
            

    def initNetwork(self): 
        super().initNetwork()
         
        self.session = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=2, allow_soft_placement=True))
        ckpt = tf.train.get_checkpoint_state(self.folder(self.rl_conf.checkpoint_dir))
        initializer = tf.random_uniform_initializer(-0.1, 0)
        self.numIterations = 0
        
        if self.start_fresh:
            print("Initializing the network from scratch", level=9)
            with tf.variable_scope("targetnet", reuse=None, initializer=initializer):
                self.target_model = self.usesnetwork(self.rl_conf, self, mode="inference", rl_not_trainables=DONT_TRAIN)                
            with tf.variable_scope("onlinenet", reuse=None, initializer=initializer):
                self.online_model = self.usesnetwork(self.rl_conf, self, mode="rl_train", rl_not_trainables=DONT_TRAIN)                
            init = tf.global_variables_initializer()
            self.session.run(init)        
            self.saver = tf.train.Saver(max_to_keep=1, var_list=get_variables("onlinenet"))
            
            self.session.run([target.assign(online) for online, target in zip(get_variables(scope="onlinenet"), get_variables(scope="targetnet"))])
            
#                for i in tf.trainable_variables():
#                    if i.name.startswith("learning"):
#                        for j in tf.trainable_variables():
#                            if (not(j.name.startswith("learning"))) and j.name[j.name.find("/"):] == i.name[i.name.find("/"):]:
#                                #self.session.run(i.assign(j));
#                                print(i.eval(session=self.session) == j.eval(session=self.session), level=10)
            
        else:
            if not (ckpt and ckpt.model_checkpoint_path):
                
                print("Initializing the network from supervised pre-training", level=9)
                self.usesnetwork(self.rl_conf, self, mode="sv_train")
                varlist = dict(zip([v.name for v in tf.trainable_variables()], tf.trainable_variables()))
                varlist = list(eraseneccessary(varlist, DONT_COPY_WEIGHTS).keys())
                print(varlist)
                
                with tf.variable_scope("targetnet", reuse=None, initializer=initializer):
                    self.target_model = self.usesnetwork(self.rl_conf, self, mode="inference", rl_not_trainables=DONT_TRAIN)                
                with tf.variable_scope("onlinenet", reuse=None, initializer=initializer):
                    self.online_model = self.usesnetwork(self.rl_conf, self, mode="rl_train", rl_not_trainables=DONT_TRAIN)                
                        
                restorevars = {}
                for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='targetnet'):
                    for j in varlist:
                        if j in i.name:
                            restorevars[i.name.replace("targetnet/","").replace(":0","")] = i
                
                print(restorevars)
                
                init = tf.global_variables_initializer()
                self.session.run(init)        
                self.pretrainsaver = tf.train.Saver(restorevars)
                sv_ckpt = tf.train.get_checkpoint_state(self.folder(self.sv_conf.checkpoint_dir))
                assert sv_ckpt and sv_ckpt.model_checkpoint_path, "I need at least a supervisedly pre-trained net!"
                self.pretrainsaver.restore(self.session, sv_ckpt.model_checkpoint_path)
                self.session.run([online.assign(target) for online, target in zip(get_variables(scope="onlinenet"), get_variables(scope="targetnet"))]) #muss so rum hier!

                self.saver = tf.train.Saver(max_to_keep=1, var_list=get_variables("onlinenet"))
                
            else:
                print("Initializing the network from the last RL-Save", level=9)
                with tf.variable_scope("targetnet", reuse=None, initializer=initializer):
                    self.target_model = self.usesnetwork(self.rl_conf, self, mode="inference", rl_not_trainables=DONT_TRAIN)
                with tf.variable_scope("onlinenet", reuse=None, initializer=initializer):
                    self.online_model = self.usesnetwork(self.rl_conf, self, mode="rl_train", rl_not_trainables=DONT_TRAIN)                                            
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
        
        max_qs = np.ones_like(max_qs)*999
        
        resetafters = list(np.ones(len(resetafters))-np.array(resetafters, dtype=int))
        #Bellman equation: Q(s,a) = r + y(max(Q(s',a')))
        #qs[np.arange(BATCHSIZE), argmactions] += learning_rate*((rewards + Q_DECAY * max_qs * (not resetafters))-qs[np.arange(BATCHSIZE), argmactions]) #so wäre es wenn wir kein ANN nutzen würden!
        #https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0
        qs = np.zeros_like(qs)
#        qs[np.arange(batchsize), argmactions] = 1 #rewards + self.rl_conf.q_decay * max_qs * (not resetafters) #wenn anschließend resettet wurde war es bspw ein wallhit und damit quasi ein final state
        
        qs[np.arange(batchsize), argmactions] = rewards + self.rl_conf.q_decay * max_qs * (not resetafters) #wenn anschließend resettet wurde war es bspw ein wallhit und damit quasi ein final state
                     
#        print(qs)   
#        time.sleep(5)
          
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
    sv_conf = config.Config()
    rl_conf = config.RL_Config()
    from server import Containers; containers = Containers(); containers.sv_conf = sv_conf; containers.rl_conf = rl_conf
    
    myAgent = Agent(sv_conf, containers, rl_conf, True)
#    myAgent.svTrain()
    

    myAgent.initNetwork() #created online_model emptily, dank start_fresh
    #jetzt ne inference q-value anschauen, dann versuchen nen bisschen off-policy rl zu lernen, dann wieder inference anschauen
    
    import read_supervised
    trackingpoints = read_supervised.TPList(sv_conf.LapFolderName, sv_conf.use_second_camera, sv_conf.msperframe, sv_conf.steering_steps, sv_conf.INCLUDE_ACCPLUSBREAK)
    
    #evaluating it BEFORE
    trackingpoints.reset_batch()
    stateBatch, _ = trackingpoints.next_batch(sv_conf, myAgent, trackingpoints.numsamples)
    ev, _, _ = myAgent.online_model.run_sv_eval(myAgent.session, myAgent, stateBatch)                       
    print("Correct inferences: %.2f%%" % (ev*100), level=10)                      

    #doing stuff
    print("Number of samples:",trackingpoints.numsamples)
    for i in range(200):
        trackingpoints.reset_batch()
        while trackingpoints.has_next(32):
            QLearnInputs = read_supervised.create_QLearnInputs_from_SVStateBatch(*trackingpoints.next_batch(sv_conf, myAgent, 32), myAgent)
            myAgent.q_learn(myAgent.online_model, *QLearnInputs, 32)
    
        if (i+1) % 10 == 0:
            myAgent.saveNet()
            
        #evaluating it AFTER
        trackingpoints.reset_batch()
        stateBatch, _ = trackingpoints.next_batch(sv_conf, myAgent, trackingpoints.numsamples)
        ev, _, _ = myAgent.online_model.run_sv_eval(myAgent.session, myAgent, stateBatch)                       
        print("Iteration: %i    Correct inferences: %.2f%%" % (i, ev*100), level=10)                  



    trackingpoints.reset_batch()
    for i in range(10):
        (conv_inputs, other_inputs), _, ArgmActions, _, _ = read_supervised.create_QLearnInputs_from_SVStateBatch(*trackingpoints.next_batch(sv_conf, myAgent, 1), myAgent)
        
        conv_inputs = np.squeeze(np.array(conv_inputs))
        other_inputs = other_inputs[0]
        
        oh, q = myAgent.online_model.run_inference(myAgent.session, conv_inputs, other_inputs)
        
        print(np.argmax(np.array(oh[0])), "  ", ArgmActions[0])
    
        print(q)
    
    
    
    time.sleep(999)
    
    
    
    
    