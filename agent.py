# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 16:11:04 2017

@author: csten_000
"""
import time
current_milli_time = lambda: int(round(time.time() * 1000))
import numpy as np
import threading
####own classes###
from myprint import myprint as print
import read_supervised
import infoscreen


###################################################################################################

class AbstractAgent(object):
    def __init__(self, containers, num, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lock = threading.Lock()
        self.isinitialized = False
        self.containers = containers        
        self.number = num
        self.isbusy = False        
        
    ##############functions that should be impemented##########
    def runInference(self, update_only_if_new):
        raise NotImplementedError
        
        
        
        
    def performNetwork(self, otherinputs, visionvec):
        print("Another ANN Inference", level=6)
        
        
        
        
    def initNetwork(self, start_fresh):
        raise NotImplementedError    

    
    #################### Helper functions#######################
    def dediscretize(self, discrete, config):
        return read_supervised.dediscretize_all(discrete, config.steering_steps, config.INCLUDE_ACCPLUSBREAK)

    def discretize(self, throttle, brake, steer, config):
        discreteSteer = read_supervised.discretize_steering(steer, config.steering_steps)
        return read_supervised.discretize_all(throttle, brake, discreteSteer, config.steering_steps, config.INCLUDE_ACCPLUSBREAK)

    def inflate_speed(self, speed, config):
        return read_supervised.inflate_speed(speed, config.speed_neurons, config.SPEED_AS_ONEHOT)
    
    def resetUnity(self):
        import server
        server.resetUnity(self.containers)

###################################################################################################        
class AbstractRLAgent(AbstractAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_random_timestamp = 0
        self.last_random_action = None
        self.repeat_random_action_for = 1000
    
    
    
    
    def addToMemory(self, otherinputs, visionvec): 
        oldstate, action = self.containers.inputval.get_previous_state()
        if oldstate is not None:
            newstate = (visionvec, otherinputs.SpeedSteer.velocity)
            reward = self.calculateReward(self.containers.inputval)
            self.containers.memory.append([oldstate, action, reward, newstate, False]) 
            print(self.dediscretize(action, self.rl_config), reward, level=6)
            print(len(self.containers.memory.memory),level=6)
            
            if self.containers.showscreen:
                infoscreen.print(self.dediscretize(action, self.rl_config), round(reward,2), round(self.cnn.calculate_value(self.session, newstate[0], newstate[1], self.sv_config.history_frame_nr)[0],2), containers= self.containers, wname="Last memory")
                infoscreen.print(str(len(self.containers.memory.memory)), containers= self.containers, wname="Memorysize")
                                    
 
        



    def dauerLearnANN(self):
        while self.containers.KeepRunning:
            self.learnANN()
        print("Learn-Thread stopped")



                
    def learnANN(self):
        raise NotImplementedError



        
    def calculateReward(self, inputval):
        progress_old = inputval.previous_otherinputs.progress
        progress_new = inputval.otherinputs.progress
        if progress_old > 90 and progress_new < 10:
            progress_new += 100
        progress = round(progress_new-progress_old,3)*20
        
        stay_on_street = abs(inputval.otherinputs.CenterDist)
        #wenn er >= 10 war und seitdem keine neue action kam, muss er >= 10 bleiben!
        
        stay_on_street = round(0 if stay_on_street < 5 else 20 if stay_on_street >= 10 else stay_on_street-5, 3)
        
        return progress-stay_on_street



    
    def randomAction(self, speed, config):
        print("Random Action!", level=6)
        if current_milli_time() - self.last_random_timestamp > self.repeat_random_action_for:
            
            action = np.random.randint(4) if config.INCLUDE_ACCPLUSBREAK else np.random.randint(3)
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
            tmp = [0]*config.steering_steps
            tmp[np.random.randint(config.steering_steps)] = 1
            steer = read_supervised.dediscretize_steer(tmp)
            
            
            self.last_random_timestamp = current_milli_time()
            self.last_random_action = (throttle, brake, steer)
        else:
            throttle, brake, steer = self.last_random_action
            
        #throttle, brake, steer = 1, 0, 0
        result = "["+str(throttle)+", "+str(brake)+", "+str(steer)+"]"
        return result, self.discretize(throttle, brake, steer, config) 

