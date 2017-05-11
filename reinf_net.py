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
#====own classes====
import supervisedcnn 
import reinforcementcnn
import read_supervised

current_milli_time = lambda: int(round(time.time() * 1000))

STANDARDRETURN = ("[0.5,0,0.0]", 42)
MEMORY_SIZE = 5000
epsilon = 0.9
BATCHSIZE = 32
Q_DECAY = 0.95
repeat_random_action_for = 1000
last_random_timestamp = 0
last_random_action = None


class ReinfNet(object):
    def __init__(self, num, config):
        self.lock = threading.Lock()
        self.isinitialized = False
        self.containers = None        
        self.number = num
        self.isbusy = False
        self.config = config
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


                def prepare_feed_dict(state):
                    feed_dict = {
                      self.cnn.inputs: np.expand_dims(np.array(state[0]), axis=0),
                      self.cnn.speed_input: np.expand_dims(np.array(read_supervised.inflate_speed(state[1], supervisedcnn.Config().speed_neurons)), axis=0)
                    }
                    return feed_dict
                               
                #learn ANN
                if len(self.containers.memory.memory) > BATCHSIZE:
                    mem = self.containers.memory.memory
                    train_states = []
                    train_q_targets = []
                    samples = np.random.permutation(len(mem))[:BATCHSIZE]
                    for i in samples:
                        
                        oldstate, action, reward, newstate = mem[i]

                        q = self.session.run(self.cnn.q, feed_dict = prepare_feed_dict(oldstate))
                        q_max = self.session.run(self.cnn.q_max, feed_dict=prepare_feed_dict(newstate))
                        
                        action = np.argmax(action)
                        
                        q[0][action] = reward + (Q_DECAY * q_max)
                        
                        train_states.append(oldstate)
                        train_q_targets.append(q[0])

                    self.session.run(self.cnn.rl_train_op, feed_dict={
                        self.cnn.inputs: np.array([curr[0] for curr in train_states]),
                        self.cnn.speed_input: np.array([read_supervised.inflate_speed(curr[1], supervisedcnn.Config().speed_neurons) for curr in train_states]),
                        self.cnn.q_targets: train_q_targets,
                    })
                    
                
                
                
                
                
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
        check, networkresult = self.cnn.run_inference(self.session, visionvec, othervecs, self.config.history_frame_nr)
        if check:
            throttle, brake, steer = read_supervised.dediscretize_all((networkresult)[0])
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
        result = "["+str(throttle)+", "+str(brake)+", "+str(steer)+"]"
        return result, [throttle, brake, steer]
              
            

    def initNetwork(self):
        with tf.Graph().as_default():    
            initializer = tf.random_uniform_initializer(-0.1, 0.1)


            with tf.name_scope("SVPreLearn"):
                with tf.variable_scope("cnnmodel", reuse=None, initializer=initializer): 
                    self.svcnn = supervisedcnn.CNN(self.config, is_training=True)
                           
            self.saver = tf.train.Saver(self.svcnn.trainvars)
            self.session = tf.Session()
            ckpt = tf.train.get_checkpoint_state(self.config.checkpoint_dir) 
            assert ckpt and ckpt.model_checkpoint_path, "I need a supervisedly pre-trained net!"
            self.saver.restore(self.session, ckpt.model_checkpoint_path)
            #TODO: den quatsch soll der nur mahcen wenn der noch kein model vom reinforcementlearn hat!                      
            
            
            with tf.name_scope("ReinfLearn"): 
                self.cnn = reinforcementcnn.CNN(self.config, initializer, is_training=True)
            init = tf.global_variables_initializer()
            self.session.run(init)
                
            print("network %s initialized" %str(self.number+1))
            self.isinitialized = True
