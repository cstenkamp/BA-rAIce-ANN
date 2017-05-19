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

current_milli_time = lambda: int(round(time.time() * 1000))

STANDARDRETURN = ("[0.5,0,0.0]", 42)
MEMORY_SIZE = 5000
epsilon = 0.5
BATCHSIZE = 32
Q_DECAY = 0.95
repeat_random_action_for = 1000
last_random_timestamp = 0
last_random_action = None
CHECKPOINTALL = 5


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
        
    
    def runANN(self, update_only_if_new):
        global epsilon
        if self.isinitialized:
            if update_only_if_new and self.containers.inputval.alreadyread:
                    return
                
            self.lock.acquire()
            try:
                self.isbusy = True 
                with self.graph.as_default(): 
    #                if self.containers.inputval.othervecs[0][0] > 30 and self.containers.inputval.othervecs[0][0] < 40:
    #                    self.containers.outputval.send_via_senderthread("pleasereset", self.containers.inputval.timestamp)
    #                    return
                    othervecs, visionvec = self.containers.inputval.read()
                    
                    #add to memory
                    oldstate, action = self.containers.inputval.get_previous_state()
                    if oldstate is not None:
                        newstate = (visionvec, othervecs[1][4])
                        reward = self.calculateReward()
                        self.containers.memory.append([oldstate, action, reward, newstate]) 
                    
                    #run ANN
                    if np.random.random() > epsilon:
                        returnstuff, original = self.performNetwork(othervecs, visionvec)
                        epsilon -= 0.005
                    else:
                        returnstuff, original = self.randomAction()
                        
                    self.containers.inputval.addResultAndBackup(original) 
                    self.containers.outputval.update(returnstuff, self.containers.inputval.timestamp)    
    
                    #learn ANN
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
                        oldstates, actions, rewards, newstates = zip(*batch)                        
                        
                        qs = self.session.run(self.cnn.q, feed_dict = prepare_feed_dict(oldstates))
                        max_qs = self.session.run(self.cnn.q_max, feed_dict=prepare_feed_dict(newstates))
                        
                        argmactions = [np.argmax(i) for i in actions]
                                                
                        qs[np.arange(BATCHSIZE), argmactions] = rewards + Q_DECAY * max_qs 

    
                        self.session.run(self.cnn.rl_train_op, feed_dict={
                            self.cnn.inputs: np.array([curr[0] for curr in oldstates]),
                            self.cnn.speed_input: np.array([read_supervised.inflate_speed(curr[1], supervisedcnn.Config().speed_neurons, supervisedcnn.Config().SPEED_AS_ONEHOT) for curr in oldstates]),
                            self.cnn.q_targets: qs,
                        })
                        
                        self.containers.reinfNetSteps += 1
                        print("ReinfLearnSteps:", self.containers.reinfNetSteps)
                        
                        if self.containers.reinfNetSteps % CHECKPOINTALL == 0:
                            checkpoint_file = os.path.join(self.rl_config.checkpoint_dir, 'model.ckpt')
                            self.saver.save(self.session, checkpoint_file, global_step=self.cnn.global_step.eval(session=self.session))       
                            print("saved")
                    
                    
                    
                    
                self.isbusy = False
            finally:
                self.lock.release()
                
                
    def calculateReward(self):
        progress_old = self.containers.inputval.previous_othervecs[0][0]
        progress_new = self.containers.inputval.othervecs[0][0]
        if progress_old > 90 and progress_new < 10:
            progress_new += 100
        progress = round(progress_new-progress_old,3)
        
        stay_on_street = abs(self.containers.inputval.othervecs[3][0])
        stay_on_street = round(0 if stay_on_street < 5 else 100 if stay_on_street > 10 else stay_on_street-5, 3)
        
        
        return progress-stay_on_street

        
                        
                

    def performNetwork(self, othervecs, visionvec):
        print("Another ANN Inference")
        check, networkresult = self.cnn.run_inference(self.session, visionvec, othervecs, self.sv_config.history_frame_nr)
        if check:
            throttle, brake, steer = read_supervised.dediscretize_all(networkresult[0], self.containers.rl_conf.steering_steps, self.containers.rl_conf.INCLUDE_ACCPLUSBREAK)
            result = "["+str(throttle)+", "+str(brake)+", "+str(steer)+"]"
            return result, [throttle, brake, steer]
        else:
            return STANDARDRETURN

            
    def randomAction(self):
        global last_random_timestamp, last_random_action
        print("Random Action!")
        if current_milli_time() - last_random_timestamp > repeat_random_action_for:
            throttle = 1 if np.random.random() > 0.5 else 0
            if throttle == 1:
                brake = 1 if np.random.random() > 0.5 else 0
            else:
                brake = 1 if np.random.random() > 0.9 else 0
            #steer = ((np.random.random()*2)-1)
            steer = np.random.normal(scale=0.5)
            last_random_timestamp = current_milli_time()
            last_random_action = (throttle, brake, steer)
        else:
            throttle, brake, steer = last_random_action
            
        throttle, brake, steer = 1, 0, 0
        result = "["+str(throttle)+", "+str(brake)+", "+str(steer)+"]"
        return result, [throttle, brake, steer]
              
            

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
                    if "FC2" in key:
                        topop.append(key)
                for i in topop:
                    pretrainvars.pop(i)
                print(pretrainvars.keys())
                    
                self.pretrainsaver = tf.train.Saver(pretrainvars)
                with tf.name_scope("ReinfLearn"): 
                    with tf.variable_scope("cnnmodel", reuse=None, initializer=initializer):
                        self.cnn = reinforcementcnn.CNN(self.rl_config, initializer, is_training=True, continuing = True)
                        
                sv_ckpt = tf.train.get_checkpoint_state(self.sv_config.checkpoint_dir) 
                assert sv_ckpt and sv_ckpt.model_checkpoint_path, "I need at least a supervisedly pre-trained net!"
                self.pretrainsaver.restore(self.session, sv_ckpt.model_checkpoint_path)
                
                init = tf.global_variables_initializer()
                self.session.run(init)
                self.saver = tf.train.Saver(max_to_keep=3)
            else:
                with tf.name_scope("ReinfLearn"): 
                    with tf.variable_scope("cnnmodel", reuse=None, initializer=initializer):
                        self.cnn = reinforcementcnn.CNN(self.rl_config, initializer, is_training=True, continuing = True)
                self.saver = tf.train.Saver(max_to_keep=3)
                self.saver.restore(self.session, ckpt.model_checkpoint_path)
                self.containers.reinfNetSteps = self.cnn.global_step.eval(session=self.session)
                                
            
            print("network %s initialized with %i iterations already run." %(str(self.number+1), self.containers.reinfNetSteps))
            self.isinitialized = True