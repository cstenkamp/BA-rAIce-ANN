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
import time
import os
#====own classes====
import supervisedcnn 
import reinforcementcnn
import read_supervised
from myprint import myprint as print
import server

current_milli_time = lambda: int(round(time.time() * 1000))

STANDARDRETURN = ("[0.5,0,0.0]", [0]*42)
MEMORY_SIZE = 5000
epsilon = 0
EPSILONDECREASE = 0.0025
minepsilon = 0
BATCHSIZE = 32
Q_DECAY = 0.8
repeat_random_action_for = 1000
last_random_timestamp = 0
last_random_action = None
CHECKPOINTALL = 5
DONT_COPY_WEIGHTS = [] #["FC1", "FC2"]

ACTION_ALL_X_MS = 0
LAST_ACTION = 0
ONLY_START = False


class ReinfNet(object):
    def __init__(self, num, sv_config, containers, rl_config):
        self.lock = threading.Lock()
        self.isinitialized = False
        self.containers = containers
        self.number = num
        self.isbusy = False
        self.sv_config = sv_config
        self.rl_config = rl_config
        #tps = read_supervised.TPList(read_supervised.FOLDERNAME, config.msperframe)
        #self.normalizers = tps.find_normalizers()
        self.initNetwork()

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
        self.containers.outputval.send_via_senderthread("pleasereset", self.containers.inputval.timestamp)
        server.resetServer(self.containers, self.containers.inputval.msperframe)
    

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
    #                if self.containers.inputval.othervecs[0][0] > 30 and self.containers.inputval.othervecs[0][0] < 40:
    #                    resetUnity()
    #                    return
                    othervecs, visionvec = self.containers.inputval.read()
                    
                    #add to memory
                    oldstate, action = self.containers.inputval.get_previous_state()
                    if oldstate is not None:
                        newstate = (visionvec, othervecs[1][4])
                        reward = self.calculateReward()
                        self.containers.memory.append([oldstate, action, reward, newstate, False]) 
                        print(self.dediscretize(action), reward, level=6)
                        #deletethispart
                        if ONLY_START:
                            self.resetUnity()
                            LAST_ACTION -= ACTION_ALL_X_MS
                            return
                    

                    #run ANN
                    if np.random.random() > epsilon:
                        returnstuff, original = self.performNetwork(othervecs, visionvec)
                        epsilon = max(epsilon-EPSILONDECREASE, minepsilon)
                    else:
                        returnstuff, original = self.randomAction()
                        
                    
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
        def prepare_feed_dict(states):
            feed_dict = {
              self.cnn.inputs: np.array([state[0] for state in states]),
              self.cnn.speed_input: np.array([read_supervised.inflate_speed(state[1], supervisedcnn.Config().speed_neurons, supervisedcnn.Config().SPEED_AS_ONEHOT) for state in states])
            }
            return feed_dict
            
            
        if len(self.containers.memory.memory) > BATCHSIZE:
        
            mem = self.containers.memory.memory
            samples = np.random.permutation(len(mem))[:BATCHSIZE]

            batch = [mem[i] for i in samples]
            oldstates, actions, rewards, newstates, resetafters = zip(*batch)                        
            
            argmactions = [np.argmax(i) for i in actions]
            
            actualActions = [read_supervised.dediscretize_all(i, self.rl_config.steering_steps, self.rl_config.INCLUDE_ACCPLUSBREAK) for i in actions]
            print(dict(zip(rewards,actualActions)), level=6)
            
            qs = self.session.run(self.cnn.q, feed_dict = prepare_feed_dict(oldstates))
            max_qs = self.session.run(self.cnn.q_max, feed_dict=prepare_feed_dict(newstates))
                                         
            qs[np.arange(BATCHSIZE), argmactions] = rewards + Q_DECAY * max_qs * (not resetafters) #wenn anschließend resettet wurde war es bspw ein wallhit und damit quasi ein final state


            self.session.run(self.cnn.rl_train_op, feed_dict={
                self.cnn.inputs: np.array([curr[0] for curr in oldstates]),
                self.cnn.speed_input: np.array([read_supervised.inflate_speed(curr[1], supervisedcnn.Config().speed_neurons, supervisedcnn.Config().SPEED_AS_ONEHOT) for curr in oldstates]),
                self.cnn.q_targets: qs,
            })
            
            self.containers.reinfNetSteps += 1
            print("ReinfLearnSteps:", self.containers.reinfNetSteps, level=6)
            
            if self.containers.reinfNetSteps % CHECKPOINTALL == 0:
                checkpoint_file = os.path.join(self.rl_config.checkpoint_dir, 'model.ckpt')
                self.saver.save(self.session, checkpoint_file, global_step=self.cnn.global_step.eval(session=self.session))       
                print("saved")
                    
                    
                    
                    
                
                
    def calculateReward(self):
        progress_old = self.containers.inputval.previous_othervecs[0][0]
        progress_new = self.containers.inputval.othervecs[0][0]
        if progress_old > 90 and progress_new < 10:
            progress_new += 100
        progress = round(progress_new-progress_old,3)
        
        stay_on_street = abs(self.containers.inputval.othervecs[3][0])
        #wenn er >= 10 war und seitdem keine neue action kam, muss er >= 10 bleiben!
        
        stay_on_street = round(0 if stay_on_street < 5 else 50 if stay_on_street >= 10 else stay_on_street-5, 3)
        
        
        return progress-stay_on_street

        
                        
                

    def performNetwork(self, othervecs, visionvec):
        print("Another ANN Inference", level=6)
        check, (networkresult,qvals) = self.cnn.run_inference(self.session, visionvec, othervecs, self.sv_config.history_frame_nr)
        if check:
            throttle, brake, steer = read_supervised.dediscretize_all(networkresult[0], self.containers.rl_conf.steering_steps, self.containers.rl_conf.INCLUDE_ACCPLUSBREAK)
            result = "["+str(throttle)+", "+str(brake)+", "+str(steer)+"]"
            print(qvals, level=6)
            return result, networkresult[0]
        else:
            return STANDARDRETURN

            
    def randomAction(self):
        global last_random_timestamp, last_random_action
        print("Random Action!", level=6)
        if current_milli_time() - last_random_timestamp > repeat_random_action_for:
            
            action = np.random.randint(4) if self.rl_config.INCLUDE_ACCPLUSBREAK else np.random.randint(3)
            if action == 0: brake, throttle = 1, 0
            if action == 1: brake, throttle = 0, 1
            if action == 2: brake, throttle = 0, 0
            if action == 3: brake, throttle = 1, 1
                   
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
              
            

    def initNetwork(self):
        self.graph = tf.Graph()
        with self.graph.as_default():    
            
            self.session = tf.Session()
            ckpt = tf.train.get_checkpoint_state(self.rl_config.checkpoint_dir) 
            initializer = tf.random_uniform_initializer(-0.1, 0.1)
            
            if not (ckpt and ckpt.model_checkpoint_path):
                

                pretrainvars = supervisedcnn.CNN(self.sv_config, is_training=True).trainvars
                topop = []
                for key, _ in pretrainvars.items():
                    for curr in DONT_COPY_WEIGHTS:
                        if curr in key:
                            topop.append(key)
                for i in topop:
                    pretrainvars.pop(i)
                print(pretrainvars.keys())
                    
                self.pretrainsaver = tf.train.Saver(pretrainvars)
                with tf.name_scope("ReinfLearn"): 
                    with tf.variable_scope("cnnmodel", reuse=None, initializer=initializer):
                        self.cnn = reinforcementcnn.CNN(self.rl_config, is_training=True)
                        
                sv_ckpt = tf.train.get_checkpoint_state(self.sv_config.checkpoint_dir) 
                assert sv_ckpt and sv_ckpt.model_checkpoint_path, "I need at least a supervisedly pre-trained net!"
                self.pretrainsaver.restore(self.session, sv_ckpt.model_checkpoint_path)
                
                init = tf.global_variables_initializer()
                self.session.run(init)
                self.saver = tf.train.Saver(max_to_keep=3)
            else:
                with tf.name_scope("ReinfLearn"): 
                    with tf.variable_scope("cnnmodel", reuse=None, initializer=initializer):
                        self.cnn = reinforcementcnn.CNN(self.rl_config, is_training=True)
                self.saver = tf.train.Saver(max_to_keep=3)
                self.saver.restore(self.session, ckpt.model_checkpoint_path)
                self.containers.reinfNetSteps = self.cnn.global_step.eval(session=self.session)
                                
            
            print("network %s initialized with %i iterations already run." %(str(self.number+1), self.containers.reinfNetSteps))
            self.isinitialized = True