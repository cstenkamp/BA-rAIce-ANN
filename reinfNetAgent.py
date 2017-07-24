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
import dqn
from myprint import myprint as print
import infoscreen

current_milli_time = lambda: int(round(time.time() * 1000))

STANDARDRETURN = ("[0.5,0,0.0]", [0]*42)
DONT_COPY_WEIGHTS = [] #["FC1", "FC2"]
DONT_TRAIN = [] #["Conv1", "Conv2", "FC1"]# ["Conv1", "Conv2"]

lastresult = None

ONLY_START = False


class ReinfNetAgent(AbstractRLAgent):
    def __init__(self, sv_conf, containers, rl_conf, start_fresh, *args, **kwargs):
        super().__init__(sv_conf, containers, rl_conf, *args, **kwargs)
        self.epsilon = self.rl_conf.startepsilon
        self.initNetwork(start_fresh)
        self.learn_which = self.online_cnn    #TODO: target network ausschalten können
   

    def runInference(self, otherinputs, visionvec):
        if self.isinitialized:
            if not self.checkIfInference():
                return
            super().preRunInference(otherinputs, visionvec)
                


#            ##############DELETETHISPART############## #to check how fast the pure socket connection, whithout ANN, is
#            self.containers.outputval.send_via_senderthread("[0, 0, 0]", self.containers.inputval.CTimestamp, self.containers.inputval.STimestamp)
#            return
#            ##############DELETETHISPART ENDE##############
            
            #run ANN
            global lastresult
            try:
                
                if self.canLearn() and np.random.random() > self.epsilon:
                    returnstuff, original = self.performNetwork(otherinputs, visionvec)
                else:
                    returnstuff, original = self.randomAction(otherinputs.SpeedSteer.velocity, self.rl_conf)
                
                    if len(self.memory) >= self.rl_conf.replaystartsize:
                        try:
                            self.epsilon = min(round(max(self.rl_conf.startepsilon-((self.rl_conf.startepsilon-self.rl_conf.minepsilon)*((self.numIterations-self.rl_conf.replaystartsize)/self.rl_conf.finalepsilonframe)), self.rl_conf.minepsilon), 5), 1)
                        except:
                            self.epsilon = min(round(max(self.epsilon-self.rl_conf.epsilondecrease, self.rl_conf.minepsilon), 5), 1)
                        
                    if self.containers.showscreen:
                        infoscreen.print(self.epsilon, containers= self.containers, wname="Epsilon")
                lastresult = returnstuff, original
            except IndexError: #kommt wenn inputval resettet wurde
                returnstuff, original = lastresult

            if self.containers.showscreen:
                infoscreen.print(returnstuff, containers= self.containers, wname="Last command")
                if self.numIterations % 100 == 0:
                    infoscreen.print(self.reinfNetSteps, "Iterations: >"+str(self.numIterations), containers= self.containers, wname="ReinfLearnSteps")

            super().postRunInference(returnstuff, original)
    

    #dauerlearnANN kommt aus der AbstractRLAgent
                
                
    def learnANN(self):   
        def prepare_feed_dict(states, which_net):
            feed_dict = {
              which_net.inputs: np.array([state[0] for state in states]),
              which_net.speed_input:  np.array([self.inflate_speed(state[1], self.rl_conf) for state in states])
            }
            return feed_dict
            
            
        batch = self.memory.sample(self.rl_conf.batchsize)
        oldstates, argmactions, rewards, newstates, resetafters = zip(*batch)      
        actions = np.zeros([len(argmactions), ((4*self.rl_conf.steering_steps) if self.rl_conf.INCLUDE_ACCPLUSBREAK else (3*self.rl_conf.steering_steps))])
        for i in range(len(argmactions)):
            actions[i][argmactions[i]] = 1
        
        
        actualActions = [self.dediscretize(i, self.rl_conf) for i in actions]
        print(list(zip(rewards,actualActions)), level=4)
        
        qs = self.session.run(self.learn_which.q, feed_dict = prepare_feed_dict(oldstates, self.learn_which)) 
        max_qs = self.session.run(self.learn_which.q_max, feed_dict=prepare_feed_dict(newstates, self.learn_which))
                                     
        #Bellman equation: Q(s,a) = r + y(max(Q(s',a')))
        #qs[np.arange(BATCHSIZE), argmactions] += learning_rate*((rewards + Q_DECAY * max_qs * (not resetafters))-qs[np.arange(BATCHSIZE), argmactions]) #so wäre es wenn wir kein ANN nutzen würden!
        #https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0
        qs[np.arange(self.rl_conf.batchsize), argmactions] = rewards + self.rl_conf.q_decay * max_qs * (not resetafters) #wenn anschließend resettet wurde war es bspw ein wallhit und damit quasi ein final state
        
        self.session.run(self.learn_which.train_op, feed_dict={
            self.learn_which.inputs: np.array([curr[0] for curr in oldstates]),
            self.learn_which.speed_input:np.array([self.inflate_speed(curr[1], self.rl_conf) for curr in oldstates]),
            self.learn_which.targets: qs,
        })
        
        self.reinfNetSteps += 1
        print("ReinfLearnSteps:", self.reinfNetSteps, level=3)
        if self.containers.showscreen:
            infoscreen.print(self.reinfNetSteps, "Iterations: >"+str(self.numIterations), containers= self.containers, wname="ReinfLearnSteps")
                    
        if self.reinfNetSteps % self.rl_conf.checkpointall == 0 or self.numIterations >= self.rl_conf.train_for:
            self.saveNet()
            
            
        if self.learn_which == self.online_cnn:
            self.lock.acquire()
            if self.reinfNetSteps % self.rl_conf.copy_target_all == 0:
                self.freezeEverything("saveNet")
                with self.graph.as_default():    
                    self.session.run([target.assign(online) for online, target in zip(get_variables(scope="onlinenet"), get_variables(scope="targetnet"))])
                if self.containers.showscreen:
                    infoscreen.print(time.strftime("%H:%M:%S", time.gmtime()), containers= self.containers, wname="Last Targetnet Copy")
                self.unFreezeEverything("saveNet")
            self.lock.release()
        

                        
                
    def saveNet(self):
        #self.freezeEverything("saveNet")
        self.target_cnn.saveNumIters(self.session, self.numIterations)
        checkpoint_file = os.path.join(self.rl_conf.checkpoint_dir, 'model.ckpt')
        self.saver.save(self.session, checkpoint_file, global_step=self.learn_which.global_step.eval(session=self.session))       
        print("saved", level=6)
        #self.unFreezeEverything("saveNet")
                
                
    #calculateReward ist in der AbstractRLAgent von der er erbt

        
                        
                

    def performNetwork(self, otherinputs, visionvec):        
        super().performNetwork(otherinputs, visionvec)
        
        with self.graph.as_default():
            check, (networkresult, qvals) = self.target_cnn.run_inference(self.session, visionvec, otherinputs)
            if check:
                throttle, brake, steer = self.dediscretize(networkresult[0], self.containers.rl_conf)
                result = "["+str(throttle)+", "+str(brake)+", "+str(steer)+"]"
                self.showqvals(qvals[0])
                return result, networkresult[0]
            else:
                return lastresult
        

    def showqvals(self, qvals):
        amount = self.rl_conf.steering_steps*4 if self.rl_conf.INCLUDE_ACCPLUSBREAK else self.rl_conf.steering_steps*3
        b = []
        for i in range(amount):
            a = [0]*amount
            a[i] = 1
            b.append(str(self.dediscretize(a, self.rl_conf)))
        b = list(zip(b, qvals))
        toprint = [str(i[0])[1:-1]+": "+str(i[1]) for i in b]
        toprint = "\n".join(toprint)
        
        print(b, level=3)
        if self.containers.showscreen:
            infoscreen.print(toprint, containers= self.containers, wname="Current Q Vals")
        
            
    #randomAction ist ebenfalls in der AbstractRLAgent     
            

    def initNetwork(self, start_fresh):
        
        self.graph = tf.Graph()
        with self.graph.as_default():    
            
            self.session = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=2, allow_soft_placement=True))
            ckpt = tf.train.get_checkpoint_state(self.rl_conf.checkpoint_dir) 
            initializer = tf.random_uniform_initializer(-0.1, 0.1)
            
            if start_fresh:
                with tf.name_scope("ReinfLearn"): 
                    with tf.variable_scope("targetnet", reuse=None, initializer=initializer):
                        self.target_cnn = dqn.CNN(self.rl_conf, mode="inference", rl_not_trainables=DONT_TRAIN)                
                    with tf.variable_scope("onlinenet", reuse=None, initializer=initializer):
                        self.online_cnn = dqn.CNN(self.rl_conf, mode="rl_train", rl_not_trainables=DONT_TRAIN)                
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
                    
                    dqn.CNN(self.rl_conf, mode="sv_train")
                    varlist = dict(zip([v.name for v in tf.trainable_variables()], tf.trainable_variables()))
                    varlist = list(eraseneccessary(varlist, DONT_COPY_WEIGHTS).keys())
                    print(varlist)
                    
                    with tf.name_scope("ReinfLearn"): 
                        with tf.variable_scope("targetnet", reuse=None, initializer=initializer):
                            self.target_cnn = dqn.CNN(self.rl_conf, mode="inference", rl_not_trainables=DONT_TRAIN)                
                        with tf.variable_scope("onlinenet", reuse=None, initializer=initializer):
                            self.online_cnn = dqn.CNN(self.rl_conf, mode="rl_train", rl_not_trainables=DONT_TRAIN)                
                            
                    restorevars = {}
                    for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='targetnet'):
                        for j in varlist:
                            if j in i.name:
                                restorevars[i.name.replace("targetnet/","").replace(":0","")] = i
                    
                    init = tf.global_variables_initializer()
                    self.session.run(init)        
                    self.pretrainsaver = tf.train.Saver(restorevars)
                    sv_ckpt = tf.train.get_checkpoint_state(self.sv_conf.checkpoint_dir) 
                    assert sv_ckpt and sv_ckpt.model_checkpoint_path, "I need at least a supervisedly pre-trained net!"
                    self.pretrainsaver.restore(self.session, sv_ckpt.model_checkpoint_path)
                    self.session.run([online.assign(target) for online, target in zip(get_variables(scope="onlinenet"), get_variables(scope="targetnet"))])
    
                    self.saver = tf.train.Saver(max_to_keep=1)
                    
                else:
                    with tf.name_scope("ReinfLearn"): 
                        with tf.variable_scope("targetnet", reuse=None, initializer=initializer):
                            self.target_cnn = dqn.CNN(self.rl_conf, mode="inference", rl_not_trainables=DONT_TRAIN)
                        with tf.variable_scope("onlinenet", reuse=None, initializer=initializer):
                            self.online_cnn = dqn.CNN(self.rl_conf, mode="rl_train", rl_not_trainables=DONT_TRAIN)                                            
                    self.saver = tf.train.Saver(max_to_keep=1)
                    self.saver.restore(self.session, ckpt.model_checkpoint_path)
                    self.session.run([online.assign(target) for online, target in zip(get_variables(scope="onlinenet"), get_variables(scope="targetnet"))])
                    self.session.run(self.target_cnn.global_step.assign(self.online_cnn.global_step))
                    self.reinfNetSteps = self.target_cnn.global_step.eval(session=self.session)
                    self.numIterations = self.target_cnn.restoreNumIters(self.session)
            
            print("network initialized with %i reinfNetSteps already run." % self.reinfNetSteps)
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
