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
import cnn
from myprint import myprint as print
import infoscreen

current_milli_time = lambda: int(round(time.time() * 1000))

STANDARDRETURN = ("[0.5,0,0.0]", [0]*42)
DONT_COPY_WEIGHTS = [] #["FC1", "FC2"]
DONT_TRAIN = [] #["Conv1", "Conv2", "FC1"]# ["Conv1", "Conv2"]

lastresult = None

ACTION_ALL_X_MS = 0
LAST_ACTION = 0
ONLY_START = False


class ReinfNetAgent(AbstractRLAgent):
    def __init__(self, sv_config, containers, rl_config, start_fresh, *args, **kwargs):
        super().__init__(containers, *args, **kwargs)
        self.sv_config = sv_config
        self.rl_config = rl_config
        self.epsilon = self.rl_config.startepsilon
        self.initNetwork(start_fresh)

   

    def runInference(self, update_only_if_new):
        if self.isinitialized:
            super().runInference(update_only_if_new)
                
            self.lock.acquire()
            try:
                self.isbusy = True 
                
                #delete this part
                if ACTION_ALL_X_MS:
                    global LAST_ACTION
                    if current_milli_time()-LAST_ACTION < ACTION_ALL_X_MS:
                        return
                    else:
                        LAST_ACTION = current_milli_time()
                                
#                if self.containers.inputval.otherinputs.progress > 0 and self.containers.inputval.otherinputs.progress < 10:
#                    self.resetUnity()
#                        return
                otherinputs, visionvec = self.containers.inputval.read()
                
                #add to memory
                if self.addToMemory(otherinputs, visionvec):
                    return

                #run ANN
                global lastresult
                try:
                    
                    if len(self.memory) >= self.rl_config.replaystartsize and np.random.random() > self.epsilon:
                        returnstuff, original = self.performNetwork(otherinputs, visionvec)
                    else:
                        returnstuff, original = self.randomAction(otherinputs.SpeedSteer.velocity, self.rl_config)
                    
                        if len(self.memory) >= self.rl_config.replaystartsize:
                            try:
                                self.epsilon = round(max(self.rl_config.startepsilon-((self.rl_config.startepsilon-self.rl_config.minepsilon)*((self.numIterations-self.rl_config.replaystartsize)/self.rl_config.finalepsilonframe)), self.rl_config.minepsilon), 5)
                            except:
                                self.epsilon = round(max(self.epsilon-self.rl_config.epsilondecrease, self.rl_config.minepsilon),5)
                            
                        if self.containers.showscreen:
                            infoscreen.print(self.epsilon, containers= self.containers, wname="Epsilon")
                    lastresult = returnstuff, original
                except IndexError: #kommt wenn inputval resettet wurde
                    returnstuff, original = lastresult

                if self.containers.showscreen:
                    infoscreen.print(returnstuff, containers= self.containers, wname="Last command")

                self.containers.inputval.addResultAndBackup(original) 
                self.containers.outputval.update(returnstuff, self.containers.inputval.timestamp)  


                #im original DQN learnt er halt jetzt direkt, aber er kann doch besser durchgehend lernen?
                
            finally:
                self.isbusy = False
                self.lock.release()
    
    
    def addToMemory(self, otherinputs, visionvec):
        super().addToMemory(otherinputs, visionvec)
        if ONLY_START:
            global LAST_ACTION
            self.resetUnity()
            LAST_ACTION -= ACTION_ALL_X_MS
            return True
   

    #dauerlearnANN kommt aus der AbstractRLAgent
                
                
    def learnANN(self):
        learn_which = self.online_cnn    #TODO: target network ausschalten können
  
        if self.numIterations < self.rl_config.replaystartsize:
            return
    
        def prepare_feed_dict(states, which_net):
            feed_dict = {
              which_net.inputs: np.array([state[0] for state in states]),
              which_net.speed_input:  np.array([self.inflate_speed(state[1], self.rl_config) for state in states])
            }
            return feed_dict
            
            
        if len(self.memory) > self.rl_config.batchsize+3: 

            batch = self.memory.sample(self.rl_config.batchsize)
            oldstates, actions, rewards, newstates, resetafters = zip(*batch)              
            
            argmactions = [np.argmax(i) for i in actions]
            
            actualActions = [self.dediscretize(i, self.rl_config) for i in actions]
            print(list(zip(rewards,actualActions)), level=4)
            
            qs = self.session.run(learn_which.q, feed_dict = prepare_feed_dict(oldstates, learn_which))
            max_qs = self.session.run(learn_which.q_max, feed_dict=prepare_feed_dict(newstates, learn_which))
                                         
            #Bellman equation: Q(s,a) = r + y(max(Q(s',a')))
            #qs[np.arange(BATCHSIZE), argmactions] += learning_rate*((rewards + Q_DECAY * max_qs * (not resetafters))-qs[np.arange(BATCHSIZE), argmactions]) #so wäre es wenn wir kein ANN nutzen würden!
            #https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0
            qs[np.arange(self.rl_config.batchsize), argmactions] = rewards + self.rl_config.q_decay * max_qs * (not resetafters) #wenn anschließend resettet wurde war es bspw ein wallhit und damit quasi ein final state
            
            self.session.run(learn_which.train_op, feed_dict={
                learn_which.inputs: np.array([curr[0] for curr in oldstates]),
                learn_which.speed_input:np.array([self.inflate_speed(curr[1], self.rl_config) for curr in oldstates]),
                learn_which.targets: qs,
            })
            
            self.containers.reinfNetSteps += 1
            print("ReinfLearnSteps:", self.containers.reinfNetSteps)
            if self.containers.showscreen:
                infoscreen.print(self.containers.reinfNetSteps, containers= self.containers, wname="ReinfLearnSteps")
            
            if self.containers.reinfNetSteps % self.rl_config.checkpointall == 0:
                self.cnn.saveNumIters(self.session, self.numIterations)
                checkpoint_file = os.path.join(self.rl_config.checkpoint_dir, 'model.ckpt')
                self.saver.save(self.session, checkpoint_file, global_step=learn_which.global_step.eval(session=self.session))       
                print("saved")
                
            if learn_which == self.online_cnn:
                self.lock.acquire()
                if self.containers.reinfNetSteps % self.rl_config.copy_target_all == 0:
                    with self.graph.as_default():    
                        self.session.run([target.assign(online) for online, target in zip(get_variables(scope="onlinenet"), get_variables(scope="targetnet"))])
                self.lock.release()
        
        if self.numIterations > self.rl_config.train_for:
            return "break"
                        
                
                
    #calculateReward ist in der AbstractRLAgent von der er erbt

        
                        
                

    def performNetwork(self, otherinputs, visionvec):        
        super().performNetwork(otherinputs, visionvec)
        
        with self.graph.as_default():
            check, (networkresult, qvals) = self.cnn.run_inference(self.session, visionvec, otherinputs, self.sv_config.history_frame_nr)
            if check:
                throttle, brake, steer = self.dediscretize(networkresult[0], self.containers.rl_conf)
                result = "["+str(throttle)+", "+str(brake)+", "+str(steer)+"]"
                self.showqvals(qvals[0])
                return result, networkresult[0]
            else:
                return lastresult
        

    def showqvals(self, qvals):
        amount = self.rl_config.steering_steps*4 if self.rl_config.INCLUDE_ACCPLUSBREAK else self.rl_config.steering_steps*3
        b = []
        for i in range(amount):
            a = [0]*amount
            a[i] = 1
            b.append(str(self.dediscretize(a, self.rl_config)))
        b = list(zip(b, qvals))
        toprint = [str(i[0])[1:-1]+": "+str(i[1]) for i in b]
        toprint = "\n".join(toprint)
        
        print(b, level=6)
        if self.containers.showscreen:
            infoscreen.print(toprint, containers= self.containers, wname="Current Q Vals")
        
            
    #randomAction ist ebenfalls in der AbstractRLAgent     
            

    def initNetwork(self, start_fresh):
        
        #TODO: self.numIterations aus dem checkpoint laden
        #TODONOW
        
        
        
        self.graph = tf.Graph()
        with self.graph.as_default():    
            
            self.session = tf.Session()
            ckpt = tf.train.get_checkpoint_state(self.rl_config.checkpoint_dir) 
            initializer = tf.random_uniform_initializer(-0.1, 0.1)
            
            if start_fresh:
                with tf.name_scope("ReinfLearn"): 
                    with tf.variable_scope("targetnet", reuse=None, initializer=initializer):
                        self.cnn = cnn.CNN(self.rl_config, is_reinforcement=True, is_training=True, rl_not_trainables=DONT_TRAIN)                
                    with tf.variable_scope("onlinenet", reuse=None, initializer=initializer):
                        self.online_cnn = cnn.CNN(self.rl_config, is_reinforcement=True, is_training=True, rl_not_trainables=DONT_TRAIN)                
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
                    
                    cnn.CNN(self.rl_config, is_reinforcement=False, is_training=True)
                    varlist = dict(zip([v.name for v in tf.trainable_variables()], tf.trainable_variables()))
                    varlist = list(eraseneccessary(varlist, DONT_COPY_WEIGHTS).keys())
                    print(varlist)
                    
                    with tf.name_scope("ReinfLearn"): 
                        with tf.variable_scope("targetnet", reuse=None, initializer=initializer):
                            self.cnn = cnn.CNN(self.rl_config, is_reinforcement=True, is_training=True, rl_not_trainables=DONT_TRAIN)                
                        with tf.variable_scope("onlinenet", reuse=None, initializer=initializer):
                            self.online_cnn = cnn.CNN(self.rl_config, is_reinforcement=True, is_training=True, rl_not_trainables=DONT_TRAIN)                
                            
                    restorevars = {}
                    for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='targetnet'):
                        for j in varlist:
                            if j in i.name:
                                restorevars[i.name.replace("targetnet/","").replace(":0","")] = i
                    
                    init = tf.global_variables_initializer()
                    self.session.run(init)        
                    self.pretrainsaver = tf.train.Saver(restorevars)
                    sv_ckpt = tf.train.get_checkpoint_state(self.sv_config.checkpoint_dir) 
                    assert sv_ckpt and sv_ckpt.model_checkpoint_path, "I need at least a supervisedly pre-trained net!"
                    self.pretrainsaver.restore(self.session, sv_ckpt.model_checkpoint_path)
                    self.session.run([online.assign(target) for online, target in zip(get_variables(scope="onlinenet"), get_variables(scope="targetnet"))])
    
                    self.saver = tf.train.Saver(max_to_keep=1)
                    
                else:
                    with tf.name_scope("ReinfLearn"): 
                        with tf.variable_scope("targetnet", reuse=None, initializer=initializer):
                            self.cnn = cnn.CNN(self.rl_config, is_reinforcement=True, is_training=True, rl_not_trainables=DONT_TRAIN)
                        with tf.variable_scope("onlinenet", reuse=None, initializer=initializer):
                            self.online_cnn = cnn.CNN(self.rl_config, is_reinforcement=True, is_training=True, rl_not_trainables=DONT_TRAIN)                                            
                    self.saver = tf.train.Saver(max_to_keep=1)
                    self.saver.restore(self.session, ckpt.model_checkpoint_path)
                    self.session.run([online.assign(target) for online, target in zip(get_variables(scope="onlinenet"), get_variables(scope="targetnet"))])
                    self.session.run(self.cnn.global_step.assign(self.online_cnn.global_step))
                    self.containers.reinfNetSteps = self.cnn.global_step.eval(session=self.session)
                    self.numIterations = self.cnn.restoreNumIters(self.session)
            
            print("network initialized with %i iterations already run." % self.containers.reinfNetSteps)
            self.isinitialized = True
            
            

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
