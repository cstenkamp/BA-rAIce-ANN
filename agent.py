# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 16:11:04 2017

@author: csten_000
"""
import time
current_milli_time = lambda: int(round(time.time() * 1000))
import numpy as np
import threading
import tensorflow as tf
import os
from copy import deepcopy
####own classes###
from myprint import myprint as print
import read_supervised
import infoscreen
from dddqn import DDDQN_model 
from evaluator import evaluator
from inefficientmemory import Memory

###################################################################################################

class AbstractAgent(object):
    def __init__(self, conf, containers, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lock = threading.Lock()
        self.isinitialized = False
        self.containers = containers  
        self.conf = conf
        self.numIterations = 0
        self.ff_inputsize = 0
        self.usesConv = True
        self.conv_stacked = True
        self.ff_stacked = False
        self.model = None
        self.graph = tf.Graph()
        self.usesnetwork = DDDQN_model
        
    ##############functions that should be impemented##########
    def checkIfInference(self):
        if self.conf.UPDATE_ONLY_IF_NEW and self.containers.inputval.alreadyread:
            return False
        return True
        
    def preRunInference(self):       
        self.numIterations += 1
    
    def postRunInference(self, toUse, toSave): #toUse will already be a prepared string, toSave will be raw.
        self.containers.outputval.update(toUse, toSave, self.containers.inputval.CTimestamp, self.containers.inputval.STimestamp)  
    
    #creates the Agents state from the real state. this is the base version, other agents may overwrite it.
    def getAgentState(self, vvec1_hist, vvec2_hist, otherinput_hist, action_hist): 
        assert self.conf.use_cameras, "You disabled cameras in the config, which is impossible for this agent!"
        conv_inputs = np.concatenate([vvec1_hist, vvec2_hist]) if vvec2_hist is not None else vvec1_hist
        other_inputs = otherinput_hist[0].SpeedSteer.velocity
        stands_inputs = otherinput_hist[0].SpeedSteer.velocity < 6
        return conv_inputs, other_inputs, stands_inputs
    
    def makeNetUsableOtherInputs(self, other_inputs): #normally, the otherinputs are stored as compact as possible. Networks may need to unpack that.
        other_inputs = self.inflate_speed(other_inputs)
        assert len(other_inputs) == self.ff_inputsize        
        return other_inputs
    
    def getAction(self, _, __, ___, action_hist):
        return action_hist[0] 
    
    def makeNetUsableAction(self, action):
        return self.discretize(*action)
        
    def performNetwork(self, *args):
        print("Another ANN Inference", level=3)
        
        
    def initNetwork(self):
        raise NotImplementedError    


    #################### Helper functions#######################
    def dediscretize(self, discrete):
        if not isinstance(discrete, (list, tuple)) and type(discrete).__module__ != np.__name__:
            val = [0]*(self.conf.steering_steps*4 if self.conf.INCLUDE_ACCPLUSBREAK else self.conf.steering_steps*3)
            val[discrete] = 1
            discrete = val
        return read_supervised.dediscretize_all(discrete, self.conf.steering_steps, self.conf.INCLUDE_ACCPLUSBREAK)

    def discretize(self, throttle, brake, steer):
        return read_supervised.discretize_all(throttle, brake, steer, self.conf.steering_steps, self.conf.INCLUDE_ACCPLUSBREAK)

    def inflate_speed(self, speed):
        return read_supervised.inflate_speed(speed, self.conf.speed_neurons, self.conf.SPEED_AS_ONEHOT, self.conf.MAXSPEED)
    
    def resetUnityAndServer(self):
        import server
        server.resetUnityAndServer(self.containers)

    def folder(self, actualfolder):
        folder = self.conf.superfolder()+self.name+"/"+actualfolder
        if not os.path.exists(folder):
            os.makedirs(folder)   
        return folder

###################################################################################################   
###################################################################################################   
###################################################################################################   
###################################################################################################  

  
class AbstractRLAgent(AbstractAgent):
    def __init__(self, conf, containers, *args, **kwargs):
        super().__init__(conf, containers, *args, **kwargs)
        if not hasattr(self, "memory"): #einige agents haben bereits eine andere memory-implmentation, die sollste nicht überschreiben
            self.memory = Memory(conf.memorysize, conf, self)
        self.repeat_random_action_for = 1000
        self.freezeInfReasons = []
        self.freezeLearnReasons = [] 
        self.wallhitPunish = 1;
        self.wrongDirPunish = 10;

        
    def initNetwork(self):    
        assert self.containers is not None, "if you init the net for a RL-run, the containers must not be None!"
        self.episode_statevals = []  #für evaluator
        self.episodes = 0 #für evaluator
        self.reinfNetSteps = 0 #für evaluator
        self.numInferencesAfterLearn = 0
        self.numLearnAfterInference = 0
        self.last_random_timestamp = 0
        self.last_random_action = None
        self.evaluator = evaluator(self.containers, self, self.containers.show_plots, self.conf.save_xml,      \
                                   ["average rewards", "average Q-vals",   "progress", "laptime"                    ], \
                                   [1,                 self.conf.MAXSPEED, 100,         self.conf.time_ends_episode ] )
    
    
    #gamestate and paststate sind jeweils vvec1_hist, vvec2_hist, otherinputs_hist, action_hist #TODO: nicht mit gamestate und paststate, direkt mit agentstate!
    def addToMemory(self, gameState, pastState): 
        assert hasattr(self, "memory") and self.memory is not None, "I don't have a memory, that's fatal."
        
        if pastState: #was nicht der Fall DIREKT nach reset oder nach start ist (nur dann ist er False)
            
            past_conv_inputs, past_other_inputs, _ = self.getAgentState(*pastState)
            s  = (past_conv_inputs, past_other_inputs)
            a  = self.getAction(*pastState)  #das (throttle, brake, steer)-tuple. 
            r = self.calculateReward(*gameState)
            conv_inputs, other_inputs, _ = self.getAgentState(*gameState)
            s2 = (conv_inputs, other_inputs)
            markovtuple = (s,a,r,s2,False)            
            
            self.memory.append(markovtuple)  
            
            print("adding to Memory:",a, r, level=4) 
            
            #values for evalation:
            stateval = self.model.statevalue(markovtuple) #TODO: gucken ob stateval richtig ist!
            self.episode_statevals.append(stateval)
            
            if self.containers.showscreen:
                infoscreen.print(a, round(r,2), round(stateval,2), self.humantakingcontrolstring, containers= self.containers, wname="Last memory")
                if len(self.memory) % 20 == 0:
                    infoscreen.print(">"+str(len(self.memory)), containers= self.containers, wname="Memorysize")
      
       
     
    def checkIfInference(self):
        if self.containers.freezeInf:
            return False
        #hier gehts darum die Inference zu freezen bis das learnen eingeholt hat. (falls update_frequency gesetzt)
        if self.conf.ForEveryInf and self.conf.ComesALearn and self.canLearn() and self.conf.learnMode == "parallel":
            if self.numLearnAfterInference == self.conf.ComesALearn and self.numInferencesAfterLearn == self.conf.ForEveryInf:
                self.numLearnAfterInference = self.numInferencesAfterLearn = 0            
                self.unFreezeLearn("updateFrequency")   
                self.unFreezeInf("updateFrequency")
             #Alle ForEveryInf inferences sollst du warten, bis ComesALearn mal in der zwischenzeit gelernt wurde.
            if self.numInferencesAfterLearn == self.conf.ForEveryInf:
                #gucke ob er in der zwischenzeit ComesALearn mal gelernt hat, wenn nein, freeze Inference
                self.unFreezeLearn("updateFrequency")      
                if self.numLearnAfterInference < self.conf.ComesALearn:
                    self.freezeInf("updateFrequency")
                    print("FREEZEINF", self.numLearnAfterInference, self.numInferencesAfterLearn, level=2)
                    return super().checkIfInference()
                self.numLearnAfterInference = 0
            self.numInferencesAfterLearn += 1
        #print(self.numLearnAfterInference, self.numInferencesAfterLearn, level=10)
        return super().checkIfInference()


    def preRunInference(self, gameState, pastState):
        self.addToMemory(gameState, pastState)
        super().preRunInference()
        

    def postRunInference(self, toUse, toSave):
        super().postRunInference(toUse, toSave)
        if self.conf.learnMode == "between":
            if self.numIterations % self.conf.ForEveryInf == 0:
                print("freezing python because after", self.numIterations, "iterations I need to learn (between)", level=2)
                self.freezeInf("LearningComes")
                self.dauerLearnANN(self.conf.ComesALearn)
                self.unFreezeInf("LearningComes")
                
                
    def runInference(self, *args, **kwargs):
        raise NotImplementedError

    def canLearn(self):
        return len(self.memory) > self.conf.batchsize+self.conf.history_frame_nr+1 and \
               len(self.memory) > self.conf.replaystartsize and self.numIterations < self.conf.train_for


    def dauerLearnANN(self, learnSteps):
        i = 0
        while self.containers.KeepRunning and self.numIterations <= self.conf.train_for and i < learnSteps:
            cando = True
            #hier gehts darum das learnen zu freezen bis die Inference eingeholt hat. (falls update_frequency gesetzt)
            if self.conf.ForEveryInf and self.conf.ComesALearn and self.conf.learnMode == "parallel":
                if self.numLearnAfterInference == self.conf.ComesALearn and self.numInferencesAfterLearn == self.conf.ForEveryInf:
                    self.numLearnAfterInference = self.numInferencesAfterLearn = 0
                    self.unFreezeLearn("updateFrequency")   
                    self.unFreezeInf("updateFrequency")
                #Alle ComesALearn sollst du warten, bis ForEveryInf mal zwischenzeitlich Inference gemacht wurde
                if self.numLearnAfterInference >= self.conf.ComesALearn:
                    self.unFreezeInf("updateFrequency") 
                    if self.numInferencesAfterLearn < self.conf.ForEveryInf and self.canLearn():
                        self.freezeLearn("updateFrequency")
                        print("FREEZELEARN", self.numLearnAfterInference, self.numInferencesAfterLearn, level=2)
                        cando = False
                    else:
                        self.numInferencesAfterLearn = 0
            if cando and not self.containers.freezeLearn and self.canLearn():
                self.learnANN()
                if self.conf.ForEveryInf and self.conf.ComesALearn and self.conf.learnMode == "parallel":
                    self.numLearnAfterInference += 1
            i += 1
                                         
        self.unFreezeInf("updateFrequency") #kann hier ruhig sein, da es eh nur unfreezed falls es aufgrund von diesem grund gefreezed war.
        if self.numIterations >= self.conf.train_for: #if you exited because you're completely done
            self.saveNet()
            print("Stopping learning because I'm done after", self.numIterations, "inferences", level=10)
        
           
    def saveNet(self):
        self.freezeEverything("saveNet")
        self.model.save()
        if self.conf.save_memory_with_checkpoint:
            self.memory.save_memory()
        print("saved", level=6)
        self.unFreezeEverything("saveNet")
           
        
            
    def learnANN(self):
        raise NotImplementedError



        
    def calculateReward(self, *gameState):
        vvec1_hist, vvec2_hist, otherinput_hist, action_hist = gameState
        stay_on_street = abs(otherinput_hist[0].CenterDist)
        stay_on_street = round(0 if stay_on_street < 5 else self.wallhitPunish if stay_on_street >= 10 else stay_on_street/10, 3)
        speed = otherinput_hist[0].SpeedSteer.speedInStreetDir / self.conf.MAXSPEED
        return speed - stay_on_street


    
    def randomAction(self, speed):
        print("Random Action!", level=6)
        if current_milli_time() - self.last_random_timestamp > self.repeat_random_action_for:
            
            action = np.random.randint(4) if self.conf.INCLUDE_ACCPLUSBREAK else np.random.randint(3)
            if action == 0: brake, throttle = 0, 1
            if action == 1: brake, throttle = 0, 0
            if action == 2: brake, throttle = 1, 0
            if action == 3: brake, throttle = 1, 1
            
            if speed < 6:
                brake, throttle = 0, 1  #wenn er nicht fährt bringt stehen nix!
            
            #alternative 1a: steer = ((np.random.random()*2)-1)
            #alternative 1b: steer = min(max(np.random.normal(scale=0.5), 1), -1)
            #für 1a und 1b:  steer = read_supervised.dediscretize_steer(read_supervised.discretize_steering(steer, self.conf.steering_steps))
            #alternative 2:
            tmp = [0]*self.conf.steering_steps
            tmp[np.random.randint(self.conf.steering_steps)] = 1
            steer = read_supervised.dediscretize_steer(tmp)
            
            
            self.last_random_timestamp = current_milli_time()
            self.last_random_action = (throttle, brake, steer)
        else:
            throttle, brake, steer = self.last_random_action
            
        #throttle, brake, steer = 1, 0, 0
        result = "["+str(throttle)+", "+str(brake)+", "+str(steer)+"]"
              
        return result, (throttle, brake, steer)  #er returned immer toUse, toSave     



    def endEpisode(self, reason, gameState):  #reasons are: turnedaround, timeover, resetserver, wallhit, rounddone
        #TODO: die ersten 2 zeilen kann auch der abstractagent schon, dann muss ich im server nicht immer nach hasattr(memory) fragen!
        self.resetUnityAndServer()
        self.episodes += 1        
        
        mem_epi_slice = self.memory.endEpisode() #bei actions, nach denen resettet wurde, soll er den folgestate nicht mehr beachten (später gucken wenn reset=true dann setze Q_DECAY auf quasi 100%)
        self.eval_episodeVals(mem_epi_slice, gameState, reason)



    def eval_episodeVals(self, mem_epi_slice, gameState, endReason):
        vvec1_hist, vvec2_hist, otherinput_hist, action_hist = gameState
        avg_rewards = round(self.memory.average_rewards(mem_epi_slice[0], mem_epi_slice[1]),3)
        avg_values = round(np.mean(np.array(self.episode_statevals)), 3)
        self.episode_statevals = []
        #other evaluation-values we need are time the agent took and percentage the agent made. However, becasue those values are not neccessarily
        #officially known to the agent (since agentstate != environmentstate), we need to take them from the environment-state
        progress = round(otherinput_hist[0].ProgressVec.Progress if endReason != "lapdone" else 100, 2)
        laptime = round(otherinput_hist[0].ProgressVec.Laptime,1)
        valid = otherinput_hist[0].ProgressVec.fValidLap
        print("Avg-r:",avg_rewards,"Avg-Q:",avg_values,"progress:",progress,"laptime:",laptime,"(valid)" if valid else "", level=8)
        if self.containers.showscreen:
                infoscreen.print("rw:", avg_rewards, "Q:", avg_values, "prg:", progress, "time:", laptime, "(v)" if valid else "", containers=self.containers, wname="Last Epsd")
        
        self.evaluator.add_episode([avg_rewards, avg_values, progress, laptime], nr=self.episodes, startMemoryEntry=mem_epi_slice[0], endMemoryEntry=mem_epi_slice[1], endIteration=self.numIterations, reinfNetSteps=self.reinfNetSteps, endEpsilon=self.epsilon)
        
                         
            
            
    def punishLastAction(self, howmuch):
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
            print("freezing Unity because",reason, level=10)
            self.containers.freezeInf = True
            self.freezeInfReasons.append(reason)
            self.containers.outputval.send_via_senderthread("pleaseFreeze", self.containers.inputval.CTimestamp, self.containers.inputval.STimestamp)        
            try:
                self.containers.outputval.freezeUnity()
            except:
                pass
        
                
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
                try: #TODO: stattdessen ne variable unity_connected ahben!
                    print("unfreezing Unity because",reason, level=10)
                    self.containers.outputval.unFreezeUnity()
                except:
                    pass
        except ValueError:
            pass #you have nothing to do if it wasnt in there anyway.                      



