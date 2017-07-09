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
    def __init__(self, containers, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lock = threading.Lock()
        self.isinitialized = False
        self.containers = containers
        self.isbusy = False        
        self.numIterations = 0
        
    ##############functions that should be impemented##########
    def runInference(self, update_only_if_new):     
        self.numIterations += 1
        if update_only_if_new and self.containers.inputval.alreadyread:
            return "return"
        
        
        
        
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
        self.memory = None
        self.reinfNetSteps = 0
        self.numInferencesAfterLearn = 0
        self.numLearnAfterInference = 0
        self.freezeInfReasons = []
        self.freezeLearnReasons = []
    
    
    def addToMemory(self, otherinputs, visionvec): 
        assert self.memory is not None, "It should be specified in server right afterwards"
        
        oldstate, action = self.containers.inputval.get_previous_state()
        if oldstate is not None:
            newstate = (visionvec, otherinputs.SpeedSteer.velocity)
            reward = self.calculateReward(self.containers.inputval)
            #print(np.all(np.all(oldstate[0][0] == newstate[0][1]), np.all(oldstate[0][1] == newstate[0][2]), np.all(oldstate[0][2] == newstate[0][3])), level=10) #this is why our efficient memory works
            self.memory.append([oldstate, action, reward, newstate, False]) 
            print(self.dediscretize(action, self.rl_config), reward, level=6)
            
            if self.containers.showscreen:
                infoscreen.print(self.dediscretize(action, self.rl_config), round(reward,2), round(self.cnn.calculate_value(self.session, newstate[0], newstate[1], self.sv_config.history_frame_nr)[0],2), containers= self.containers, wname="Last memory")
                if len(self.memory) % 20 == 0:
                    infoscreen.print(">"+str(len(self.memory)), containers= self.containers, wname="Memorysize")
                                    
 
    def runInference(self, update_only_if_new):
        if self.containers.freezeInf:
            return "return"
                
        #hier gehts darum die Inference zu freezen bis das learnen eingeholt hat. (falls update_frequency gesetzt)
        if self.rl_config.ForEveryInf and self.rl_config.ComesALearn and self.canLearn():
            
            if self.numLearnAfterInference == self.rl_config.ComesALearn and self.numInferencesAfterLearn == self.rl_config.ForEveryInf:
                self.numLearnAfterInference = self.numInferencesAfterLearn = 0            
                self.unFreezeLearn("updateFrequency")   
                self.unFreezeInf("updateFrequency")
            
             #Alle ForEveryInf inferences sollst du warten, bis ComesALearn mal in der zwischenzeit gelernt wurde.
            if self.numInferencesAfterLearn == self.rl_config.ForEveryInf:
                #gucke ob er in der zwischenzeit ComesALearn mal gelernt hat, wenn nein, freeze Inference
                self.unFreezeLearn("updateFrequency")                  
           
                if self.numLearnAfterInference < self.rl_config.ComesALearn:
                    self.freezeInf("updateFrequency")
                    print("FREEZEINF", self.numLearnAfterInference, self.numInferencesAfterLearn, level=2)
                    return "return"
                self.numLearnAfterInference = 0
                
            self.numInferencesAfterLearn += 1
        
        #print(self.numLearnAfterInference, self.numInferencesAfterLearn, level=10)
        
        return super().runInference(update_only_if_new)



    def canLearn(self):
        return len(self.memory) > self.rl_config.batchsize+3 and len(self.memory) > self.rl_config.replaystartsize and self.numIterations < self.rl_config.train_for



    def dauerLearnANN(self):
        try: 
            self.rl_config.train_for
        except:
            self.rl_config.train_for = float("inf")
            
        while self.containers.KeepRunning and self.numIterations <= self.rl_config.train_for:
            cando = True
            
            #hier gehts darum das learnen zu freezen bis die Inference eingeholt hat. (falls update_frequency gesetzt)
            if self.rl_config.ForEveryInf and self.rl_config.ComesALearn:

                if self.numLearnAfterInference == self.rl_config.ComesALearn and self.numInferencesAfterLearn == self.rl_config.ForEveryInf:
                    self.numLearnAfterInference = self.numInferencesAfterLearn = 0
                    self.unFreezeLearn("updateFrequency")   
                    self.unFreezeInf("updateFrequency")
                
                #Alle ComesALearn sollst du warten, bis ForEveryInf mal zwischenzeitlich Inference gemacht wurde
                if self.numLearnAfterInference >= self.rl_config.ComesALearn:
                    self.unFreezeInf("updateFrequency") 
                    
                    if self.numInferencesAfterLearn < self.rl_config.ForEveryInf and self.canLearn():
                        self.freezeLearn("updateFrequency")
                        print("FREEZELEARN", self.numLearnAfterInference, self.numInferencesAfterLearn, level=2)
                        cando = False
                    else:
                        self.numInferencesAfterLearn = 0
        
            if cando and not self.containers.freezeLearn and self.canLearn():
                self.learnANN()
                if self.rl_config.ForEveryInf and self.rl_config.ComesALearn:
                    self.numLearnAfterInference += 1
                     
                
                    
        self.unFreezeInf("updateFrequency")
        if self.numIterations >= self.rl_config.train_for:
            self.saveNet()
        print("Stopping learning after", self.numIterations, "inferences", level=10)
        

    def saveNet(self):
        raise NotImplementedError
                
    def learnANN(self):
        raise NotImplementedError



        
    def calculateReward(self, inputval):
        progress_old = inputval.previous_otherinputs.progress
        progress_new = inputval.otherinputs.progress
        if progress_old > 90 and progress_new < 10:
            progress_new += 100
        progress = round(progress_new-progress_old,3)*100
        
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



    def endEpisode(self):
        #bei actions, nach denen resettet wurde, soll er den folgestate nicht mehr beachten (später gucken wenn reset=true dann setze Q_DECAY auf quasi 100%)
        if not self.containers.play_only:
            self.memory.endEpisode()
            
            
    def punishLastAction(self, howmuch):
        if not self.containers.play_only:
            if self.containers.showscreen:
                infoscreen.print(str(-abs(howmuch)), time.strftime("%H:%M:%S", time.gmtime()), containers=self.containers, wname="Last big punish")
            self.memory.punishLastAction(howmuch)
            
                
    
    def freezeEverything(self, reason):
        self.freezeLearn(reason)
        self.freezeInf(reason)

    def freezeLearn(self, reason):
        if not reason in self.freezeLearnReasons:
            self.containers.freezeLearn = True
            self.freezeLearnReasons.append(reason)

    def freezeInf(self, reason):
        if not reason in self.freezeInfReasons:
            self.containers.freezeInf = True
            self.freezeInfReasons.append(reason)
            self.containers.outputval.send_via_senderthread("pleaseFreeze", self.containers.inputval.timestamp)        
        
        
    def unFreezeEverything(self, reason):
        self.unFreezeLearn(reason)
        self.unFreezeInf(reason)
                
    def unFreezeLearn(self, reason):
        try:
            del self.freezeLearnReasons[self.freezeLearnReasons.index(reason)] 
            if len(self.freezeLearnReasons) == 0:
                self.containers.freezeLearn = False
        except ValueError:
            pass #you have nothing to do if it wasnt in there anyway.         

    def unFreezeInf(self, reason):
        try:
            del self.freezeInfReasons[self.freezeInfReasons.index(reason)] 
            if len(self.freezeInfReasons) == 0:
                self.containers.freezeInf = False
                self.containers.outputval.send_via_senderthread("pleaseUnFreeze", self.containers.inputval.timestamp)
        except ValueError:
            pass #you have nothing to do if it wasnt in there anyway.                      

###############################################################################
