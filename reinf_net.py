# -*- coding: utf-8 -*-
"""
Created on Wed May 10 13:33:46 2017

@author: nivradmin
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 10 13:31:54 2017

@author: nivradmin
"""

import threading
import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework import get_variables
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#====own classes====
import cnn
import read_supervised
from myprint import myprint as print
import server
import infoscreen

current_milli_time = lambda: int(round(time.time() * 1000))

STANDARDRETURN = ("[0.5,0,0.0]", [0]*42)
MEMORY_SIZE = 5000
epsilon = 0.05
EPSILONDECREASE = 0.00005
minepsilon = 0.005
BATCHSIZE = 32
Q_DECAY = 0.99
repeat_random_action_for = 1000
last_random_timestamp = 0
last_random_action = None
CHECKPOINTALL = 100
DONT_COPY_WEIGHTS = [] #["FC1", "FC2"]
DONT_TRAIN = []# ["Conv1", "Conv2"]
COPY_TARGET_ALL = 100

lastresult = None

ACTION_ALL_X_MS = 0
LAST_ACTION = 0
ONLY_START = False


class ReinfNet(object):
    def __init__(self, num, sv_config, containers, rl_config, start_fresh):
        self.lock = threading.Lock()
        self.isinitialized = False
        self.containers = containers
        self.number = num
        self.isbusy = False
        self.sv_config = sv_config
        self.rl_config = rl_config
        #tps = read_supervised.TPList(read_supervised.FOLDERNAME, config.msperframe)
        #self.normalizers = tps.find_normalizers()
        self.initNetwork(start_fresh)

#    @staticmethod
#    def flatten_oneDs(AllOneDs):
#        return np.array(read_supervised.flatten(AllOneDs))
#    
#    @staticmethod
#    def normalize_oneDs(FlatOneDs, normalizers):
#        FlatOneDs -= np.array([item[0] for item in normalizers])
#        NormalizedOneDs = FlatOneDs / np.array([item[1] for item in normalizers])
#        return NormalizedOneDs
            
    def resetUnity(self):
        server.resetUnity(self.containers)
    

    def dediscretize(self, discrete):
        return read_supervised.dediscretize_all(discrete, self.rl_config.steering_steps, self.rl_config.INCLUDE_ACCPLUSBREAK)

    def runANN(self, update_only_if_new):
        global epsilon
        if self.isinitialized:
            if update_only_if_new and self.containers.inputval.alreadyread:
                    return
                
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
                                
                with self.graph.as_default(): 
#                    if self.containers.inputval.othervecs[0][0] > 0 and self.containers.inputval.othervecs[0][0] < 10:
#                        self.resetUnity()
#                        return
                    othervecs, visionvec = self.containers.inputval.read()
                    
                    #add to memory
                    oldstate, action = self.containers.inputval.get_previous_state()
                    if oldstate is not None:
                        newstate = (visionvec, othervecs[1][4])
                        reward = self.calculateReward()
                        self.containers.memory.append([oldstate, action, reward, newstate, False]) 
                        print(self.dediscretize(action), reward, level=6)
                        print(len(self.containers.memory.memory),level=6)
                        
                        if self.containers.showscreen:
                            infoscreen.print(self.dediscretize(action), round(reward,2), round(self.cnn.calculate_value(self.session, newstate[0], newstate[1], self.sv_config.history_frame_nr)[0],2), containers= self.containers, wname="Last memory")
                            infoscreen.print(str(len(self.containers.memory.memory)), containers= self.containers, wname="Memorysize")
                        
                        #deletethispart
                        if ONLY_START:
                            self.resetUnity()
                            LAST_ACTION -= ACTION_ALL_X_MS
                            return
                    
                    #run ANN
                    global lastresult
                    try:
                        if np.random.random() > epsilon:
                            returnstuff, original = self.performNetwork(othervecs, visionvec)
                        else:
                            returnstuff, original = self.randomAction(othervecs[1][4])
                            epsilon = round(max(epsilon-EPSILONDECREASE, minepsilon),5)
                            if self.containers.showscreen:
                                infoscreen.print(epsilon, containers= self.containers, wname="Epsilon")
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
    
                
    def dauerLearnANN(self):
        while self.containers.KeepRunning:
            self.learnANN()
        print("Learn-Thread stopped")
                
                
    def learnANN(self):
        learn_which = self.online_cnn    #TODO: target network ausschalten können
    
        def prepare_feed_dict(states, which_net):
            feed_dict = {
              which_net.inputs: np.array([state[0] for state in states]),
              which_net.speed_input: np.array([read_supervised.inflate_speed(state[1], self.rl_config.speed_neurons, self.rl_config.SPEED_AS_ONEHOT) for state in states])
            }
            return feed_dict
            
            
        if len(self.containers.memory.memory) > BATCHSIZE:
        
            mem = self.containers.memory.memory
            samples = np.random.permutation(len(mem))[:BATCHSIZE]

            batch = [mem[i] for i in samples]
            oldstates, actions, rewards, newstates, resetafters = zip(*batch)                        
                       
            
            argmactions = [np.argmax(i) for i in actions]
            
            actualActions = [read_supervised.dediscretize_all(i, self.rl_config.steering_steps, self.rl_config.INCLUDE_ACCPLUSBREAK) for i in actions]
            print(list(zip(rewards,actualActions)), level=4)
            
            qs = self.session.run(learn_which.q, feed_dict = prepare_feed_dict(oldstates, learn_which))
            max_qs = self.session.run(learn_which.q_max, feed_dict=prepare_feed_dict(newstates, learn_which))
                                         
            #Bellman equation: Q(s,a) = r + y(max(Q(s',a')))
            #qs[np.arange(BATCHSIZE), argmactions] += learning_rate*((rewards + Q_DECAY * max_qs * (not resetafters))-qs[np.arange(BATCHSIZE), argmactions]) #so wäre es wenn wir kein ANN nutzen würden!
            #https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0
            qs[np.arange(BATCHSIZE), argmactions] = rewards + Q_DECAY * max_qs * (not resetafters) #wenn anschließend resettet wurde war es bspw ein wallhit und damit quasi ein final state
            
            self.session.run(learn_which.train_op, feed_dict={
                learn_which.inputs: np.array([curr[0] for curr in oldstates]),
                learn_which.speed_input: np.array([read_supervised.inflate_speed(curr[1], self.rl_config.speed_neurons, self.rl_config.SPEED_AS_ONEHOT) for curr in oldstates]),
                learn_which.targets: qs,
            })
            
            self.containers.reinfNetSteps += 1
            print("ReinfLearnSteps:", self.containers.reinfNetSteps, level=10)
            
            if self.containers.reinfNetSteps % CHECKPOINTALL == 0:
                checkpoint_file = os.path.join(self.rl_config.checkpoint_dir, 'model.ckpt')
                self.saver.save(self.session, checkpoint_file, global_step=learn_which.global_step.eval(session=self.session))       
                print("saved")
                
            if learn_which == self.online_cnn:
                self.lock.acquire()
                if self.containers.reinfNetSteps % COPY_TARGET_ALL == 0:
                    with self.graph.as_default():    
                        self.session.run([target.assign(online) for online, target in zip(get_variables(scope="onlinenet"), get_variables(scope="targetnet"))])
                self.lock.release()
                    
                
                
    def calculateReward(self):
        progress_old = self.containers.inputval.previous_othervecs[0][0]
        progress_new = self.containers.inputval.othervecs[0][0]
        if progress_old > 90 and progress_new < 10:
            progress_new += 100
        progress = round(progress_new-progress_old,3)*20
        
        stay_on_street = abs(self.containers.inputval.othervecs[3][0])
        #wenn er >= 10 war und seitdem keine neue action kam, muss er >= 10 bleiben!
        
        stay_on_street = round(0 if stay_on_street < 5 else 20 if stay_on_street >= 10 else stay_on_street-5, 3)
        
        
        return progress-stay_on_street

        
                        
                

    def performNetwork(self, othervecs, visionvec):
        print("Another ANN Inference", level=6)
        
        check, (networkresult, qvals) = self.cnn.run_inference(self.session, visionvec, othervecs, self.sv_config.history_frame_nr)
        if check:
            throttle, brake, steer = read_supervised.dediscretize_all(networkresult[0], self.containers.rl_conf.steering_steps, self.containers.rl_conf.INCLUDE_ACCPLUSBREAK)
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
            b.append(str(self.dediscretize(a)))
        b = list(zip(b, qvals))
        toprint = [str(i[0])[1:-1]+": "+str(i[1]) for i in b]
        toprint = "\n".join(toprint)
        
        print(b, level=6)
        if self.containers.showscreen:
            infoscreen.print(toprint, containers= self.containers, wname="Current Q Vals")
        
            
    def randomAction(self, speed):
        global last_random_timestamp, last_random_action
        print("Random Action!", level=6)
        if current_milli_time() - last_random_timestamp > repeat_random_action_for:
            
            
            action = np.random.randint(4) if self.rl_config.INCLUDE_ACCPLUSBREAK else np.random.randint(3)
            if action == 0: brake, throttle = 0, 1
            if action == 1: brake, throttle = 0, 0
            if action == 2: brake, throttle = 1, 0
            if action == 3: brake, throttle = 1, 1
            
            if speed < 1:
                brake, throttle = 0, 1  #wenn er nicht fährt bringt stehen nix!
            
            #alternative 1a: steer = ((np.random.random()*2)-1)
            #alternative 1b: steer = min(max(np.random.normal(scale=0.5), 1), -1)
            #für 1a und 1b:  steer = read_supervised.dediscretize_steer(read_supervised.discretize_steering(steer, self.rl_config.steering_steps))
            #alternative 2:
            tmp = [0]*self.rl_config.steering_steps
            tmp[np.random.randint(self.rl_config.steering_steps)] = 1
            steer = read_supervised.dediscretize_steer(tmp)
            
            
            last_random_timestamp = current_milli_time()
            last_random_action = (throttle, brake, steer)
        else:
            throttle, brake, steer = last_random_action
            
        #throttle, brake, steer = 1, 0, 0
        result = "["+str(throttle)+", "+str(brake)+", "+str(steer)+"]"
        return result, read_supervised.discretize_all(throttle, brake, read_supervised.discretize_steering(steer, self.rl_config.steering_steps), self.rl_config.steering_steps, self.rl_config.INCLUDE_ACCPLUSBREAK) 
              
            

    def initNetwork(self, start_fresh):
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
                self.saver = tf.train.Saver(max_to_keep=3)
                
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
    
                    self.saver = tf.train.Saver(max_to_keep=3)
                    
                else:
                    with tf.name_scope("ReinfLearn"): 
                        with tf.variable_scope("targetnet", reuse=None, initializer=initializer):
                            self.cnn = cnn.CNN(self.rl_config, is_reinforcement=True, is_training=True, rl_not_trainables=DONT_TRAIN)
                        with tf.variable_scope("onlinenet", reuse=None, initializer=initializer):
                            self.online_cnn = cnn.CNN(self.rl_config, is_reinforcement=True, is_training=True, rl_not_trainables=DONT_TRAIN)                                            
                    self.saver = tf.train.Saver(max_to_keep=3)
                    self.saver.restore(self.session, ckpt.model_checkpoint_path)
                    self.containers.reinfNetSteps = self.cnn.global_step.eval(session=self.session)
                    self.session.run([online.assign(target) for online, target in zip(get_variables(scope="onlinenet"), get_variables(scope="targetnet"))])
            
            print("network %s initialized with %i iterations already run." %(str(self.number+1), self.containers.reinfNetSteps))
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
