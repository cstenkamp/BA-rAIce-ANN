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
        self.name = __file__[__file__.rfind("\\")+1:__file__.rfind(".")]
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
        onehot, qvals = self.target_cnn.run_inference(self.session, conv_inputs, inflated_other_inputs, stands_inputs) #former is argmax, latter are individual qvals
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
                
                
    def learnANN(self):   
        batch = self.memory.sample(self.rl_conf.batchsize)
        
        if self.SAVE_ACTION_AS_ARGMAX: 
            oldstates, argmactions, rewards, newstates, resetafters = zip(*batch)      
            actions = np.zeros([len(argmactions), ((4*self.rl_conf.steering_steps) if self.rl_conf.INCLUDE_ACCPLUSBREAK else (3*self.rl_conf.steering_steps))])
            for i in range(len(argmactions)):
                actions[i][argmactions[i]] = 1
                actualActions = [self.dediscretize(i) for i in actions]
        else:
            oldstates, actualActions, rewards, newstates, resetafters = zip(*batch)      
            actualActions = [self.memory.make_floats_from_long(i) for i in actualActions]
            actions = [self.discretize(throttle, brake, steer) for throttle, brake, steer in actualActions]
            argmactions = [np.argmax(i) for i in actions]
        #soooo, actions[x] = [0,0,1,0,0], argmactions[x] = 2, actualAcitions[x] = (1,0,0)
        
        old_convs = np.array([i[0] for i in oldstates])
        old_other = np.array([self.makeNetUsableOtherInputs(i[1]) for i in oldstates])
        new_convs = np.array([i[0] for i in newstates])
        new_other = np.array([self.makeNetUsableOtherInputs(i[1]) for i in newstates])

        qs, max_qs = self.learn_which.rl_learn_forward(self.session, old_convs, old_other, new_convs, new_other)
  
        #Bellman equation: Q(s,a) = r + y(max(Q(s',a')))
        #qs[np.arange(BATCHSIZE), argmactions] += learning_rate*((rewards + Q_DECAY * max_qs * (not resetafters))-qs[np.arange(BATCHSIZE), argmactions]) #so wäre es wenn wir kein ANN nutzen würden!
        #https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0
        qs[np.arange(self.rl_conf.batchsize), argmactions] = rewards + self.rl_conf.q_decay * max_qs * (not resetafters) #wenn anschließend resettet wurde war es bspw ein wallhit und damit quasi ein final state
                   
        self.learn_which.rl_learn_step(self.session, old_convs, old_other, qs)
           
        self.reinfNetSteps += 1
        print("ReinfLearnSteps:", self.reinfNetSteps, level=3)
        if self.containers.showscreen:
            infoscreen.print(self.reinfNetSteps, "Iterations: >"+str(self.numIterations), containers= self.containers, wname="ReinfLearnSteps")
                    
        if self.reinfNetSteps % self.rl_conf.checkpointall == 0 or self.numIterations >= self.rl_conf.train_for:
            self.saveNet()
            
            
        if self.learn_which == self.online_cnn:
            if self.reinfNetSteps % self.rl_conf.copy_target_all == 0:
                self.lock.acquire()
                self.freezeEverything("saveNet")
                with self.graph.as_default():    
                    self.session.run([target.assign(online) for online, target in zip(get_variables(scope="onlinenet"), get_variables(scope="targetnet"))]) #TODO: wenns ewig lnicht funktioniert, das hier mal umdrehen
                    self.session.run(self.target_cnn.global_step.assign(self.online_cnn.global_step)) 
                if self.containers.showscreen:
                    infoscreen.print(time.strftime("%H:%M:%S", time.gmtime()), containers= self.containers, wname="Last Targetnet Copy")
                self.unFreezeEverything("saveNet")
                self.lock.release()
        
                self.evaluator.add_targetnetcopy(ReinfNetSteps=self.reinfNetSteps, MemNum=self.memory._pointer, Iterations=self.numIterations, Episode=self.episodes)

                        
                
    def saveNet(self):
        self.freezeEverything("saveNet")
        self.online_cnn.saveNumIters(self.session, self.numIterations)
        checkpoint_file = os.path.join(self.folder(self.rl_conf.checkpoint_dir), 'model.ckpt')
        self.saver.save(self.session, checkpoint_file, global_step=self.online_cnn.global_step)       
        if self.rl_conf.save_memory_with_checkpoint:
            self.memory.save_memory()
        print("saved", level=6)
        self.unFreezeEverything("saveNet")
                
                
    #calculateReward ist in der AbstractRLAgent von der er erbt

        
            
    #randomAction ist ebenfalls in der AbstractRLAgent     
            

    def initNetwork(self): 
        super().initNetwork()
         
        self.session = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=2, allow_soft_placement=True))
        ckpt = tf.train.get_checkpoint_state(self.folder(self.rl_conf.checkpoint_dir))
        initializer = tf.random_uniform_initializer(-0.1, 0.1)
        
        if self.start_fresh:
            with tf.variable_scope("targetnet", reuse=None, initializer=initializer):
                self.target_cnn = self.network(self.rl_conf, self, mode="inference", rl_not_trainables=DONT_TRAIN)                
            with tf.variable_scope("onlinenet", reuse=None, initializer=initializer):
                self.online_cnn = self.network(self.rl_conf, self, mode="rl_train", rl_not_trainables=DONT_TRAIN)                
            init = tf.global_variables_initializer()
            self.session.run(init)        
            self.saver = tf.train.Saver(max_to_keep=1)
            
            self.session.run([target.assign(online) for online, target in zip(get_variables(scope="onlinenet"), get_variables(scope="targetnet"))])
            
#                for i in tf.trainable_variables():
#                    if i.name.startswith("learning"):
#                        for j in tf.trainable_variables():
#                            if (not(j.name.startswith("learning"))) and j.name[j.name.find("/"):] == i.name[i.name.find("/"):]:
#                                #self.session.run(i.assign(j));
#                                print(i.eval(session=self.session) == j.eval(session=self.session), level=10)
            
        else:
            if not (ckpt and ckpt.model_checkpoint_path):
    
                self.network(self.rl_conf, self, mode="sv_train")
                varlist = dict(zip([v.name for v in tf.trainable_variables()], tf.trainable_variables()))
                varlist = list(eraseneccessary(varlist, DONT_COPY_WEIGHTS).keys())
                print(varlist)
                
                with tf.variable_scope("targetnet", reuse=None, initializer=initializer):
                    self.target_cnn = self.network(self.rl_conf, self, mode="inference", rl_not_trainables=DONT_TRAIN)                
                with tf.variable_scope("onlinenet", reuse=None, initializer=initializer):
                    self.online_cnn = self.network(self.rl_conf, self, mode="rl_train", rl_not_trainables=DONT_TRAIN)                
                        
                restorevars = {}
                for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='targetnet'):
                    for j in varlist:
                        if j in i.name:
                            restorevars[i.name.replace("targetnet/","").replace(":0","")] = i
                
                init = tf.global_variables_initializer()
                self.session.run(init)        
                self.pretrainsaver = tf.train.Saver(restorevars)
                sv_ckpt = tf.train.get_checkpoint_state(self.folder(self.sv_conf.checkpoint_dir))
                assert sv_ckpt and sv_ckpt.model_checkpoint_path, "I need at least a supervisedly pre-trained net!"
                self.pretrainsaver.restore(self.session, sv_ckpt.model_checkpoint_path)
                self.session.run([online.assign(target) for online, target in zip(get_variables(scope="onlinenet"), get_variables(scope="targetnet"))]) #muss so rum hier!

                self.saver = tf.train.Saver(max_to_keep=1)
                
            else:
                with tf.variable_scope("targetnet", reuse=None, initializer=initializer):
                    self.target_cnn = self.network(self.rl_conf, self, mode="inference", rl_not_trainables=DONT_TRAIN)
                with tf.variable_scope("onlinenet", reuse=None, initializer=initializer):
                    self.online_cnn = self.network(self.rl_conf, self, mode="rl_train", rl_not_trainables=DONT_TRAIN)                                            
                self.saver = tf.train.Saver(max_to_keep=1)
                self.saver.restore(self.session, ckpt.model_checkpoint_path)
                self.session.run([online.assign(target) for online, target in zip(get_variables(scope="onlinenet"), get_variables(scope="targetnet"))])
                self.reinfNetSteps = self.online_cnn.global_step.eval(session=self.session)
                self.numIterations = self.online_cnn.restoreNumIters(self.session)
        
        print("network initialized with %i reinfNetSteps already run." % self.reinfNetSteps)
        self.isinitialized = True
        self.learn_which = self.online_cnn  
            
            
            
            

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
    agent = Agent(sv_conf, None, rl_conf, True)
    agent.svTrain()