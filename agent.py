# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 16:11:04 2017

@author: csten_000
"""
import time
current_milli_time = lambda: int(round(time.time() * 1000))
import numpy as np
import threading
import os
####own classes###
from myprint import myprint as print
import read_supervised
from evaluator import evaluator
from inefficientmemory import Memory
from utils import random_unlike
from collections import deque

flatten = lambda l: [item for sublist in l for item in sublist]

###################################################################################################

class AbstractAgent(object):
    def __init__(self, conf, containers, *args, **kwargs):
        super().__init__()
        self.lock = threading.Lock()
        self.isinitialized = False
        self.containers = containers  
        self.conf = conf
        self.action_repeat = self.conf.action_repeat #kann überschrieben werden
        self.isSupervised = False #wird überschrieben
        self.isContinuous = False #wird überschrieben
        self.ff_inputsize = 0     #wird überschrieben
        self.usesConv = True      #wird überschrieben
        self.conv_stacked = True  #wird überschrieben
        self.ff_stacked = False   #wird überschrieben
        self.model = None         #wird überschrieben
    
    ###########################################################################    
    #################### Necessary functions ##################################
    ###########################################################################
    
    def checkIfAction(self):
        if self.conf.UPDATE_ONLY_IF_NEW and self.containers.inputval.alreadyread:
            return False
        return True
                
    ###########################################################################
    ################functions that may be overwritten##########################
    ###########################################################################
    
    #creates the Agents state from the real state. this is the base version, other agents may overwrite it.
    def getAgentState(self, *gameState): 
        vvec1_hist, vvec2_hist, otherinput_hist, action_hist = gameState
        assert self.conf.use_cameras, "You disabled cameras in the config, which is impossible for this agent!"
        conv_inputs = np.concatenate([vvec1_hist, vvec2_hist]) if vvec2_hist is not None else vvec1_hist
        other_inputs = [otherinput_hist[0].SpeedSteer.velocity, action_hist]
        stands_inputs = otherinput_hist[0].SpeedSteer.velocity < 10
        return conv_inputs, other_inputs, stands_inputs
    
    def makeNetUsableOtherInputs(self, other_inputs): #normally, the otherinputs are stored as compact as possible. Networks may need to unpack that.
        speed = self.inflate_speed(other_inputs[0])
        flat_actions = flatten([i if i is not None else (0,0,0) for i in other_inputs[1]])
        other_inputs = speed; other_inputs.extend(flat_actions)
        assert len(other_inputs) == self.ff_inputsize        
        return other_inputs
    
    def getAction(self, _, __, ___, action_hist):
        return action_hist[0] 
    
    def makeNetUsableAction(self, action):
        return np.argmax(self.discretize(*action))
        
    #state is either (s,a,r,s2,False) or only s. what needs to be done is make everything an array, and make action & otherinputs netusable
    def makeInferenceUsable(self, state):
        visionroll = lambda vision: np.rollaxis(vision, 0, 3) if vision is not None else None 
        makestate = lambda s: (visionroll(s[0]), self.makeNetUsableOtherInputs(s[1])) if len(s) < 3 else (visionroll(s[0]), self.makeNetUsableOtherInputs(s[1]), s[2])
        try:
            s, a, r, s2, t = state  
            s = makestate(s)
            a = self.makeNetUsableAction(a)
            s2 = makestate(s2)
            return ([s], [a], [r], [s2], [t])
        except ValueError: #too many values to unpack
            return [makestate(state)] 
                    
                    
    def initForDriving(self, *args, **kwargs):
        self.show_plots = kwargs["show_plots"] if "show_plots" in kwargs else True 
        self.numsteps = 0
        self.last_action = None
        self.repeated_action_for = self.action_repeat
        self.episode_statevals = []  #für evaluator
        self.episodes = 0 #für evaluator, wird bei jedem neustart auf null gesetzt aber das ist ok dafür
        self.evaluator = evaluator(self.containers, self, self.show_plots, self.conf.save_xml,      \
                                   ["average rewards", "average Q-vals", "progress", "laptime"                    ], \
                                   [(-0.1,1.3),        (-1,60),           100,        self.conf.time_ends_episode ] )                     
        

        
    def performAction(self, gameState, pastState):
        if self.isinitialized and self.checkIfAction():
            self.numsteps += 1
            self.repeated_action_for += 1                
            if self.repeated_action_for < self.action_repeat:
                toUse, toSave = self.last_action 
            else:
                agentState = self.getAgentState(*gameState) #may be overridden
                toUse, toSave = self.policyAction(agentState)
                self.last_action = toUse, toSave                
            self.containers.outputval.update(toUse, toSave, self.containers.inputval.CTimestamp, self.containers.inputval.STimestamp)  

                    
            
    ###########################################################################
    ################functions that should be impemented########################
    ###########################################################################
            
        
    def preTrain(self, *args, **kwargs):
        raise NotImplementedError    
        
    def policyAction(self, agentState):
        raise NotImplementedError
        
    ###########################################################################
    ########################### Helper functions###############################
    ###########################################################################
    
    def dediscretize(self, discrete):
        if not hasattr(discrete, "__len__"):  #lists, tuples and np arrays have lens, scalars (including numpy scalars) don't.
            val = [0]*(self.conf.steering_steps*4 if self.conf.INCLUDE_ACCPLUSBREAK else self.conf.steering_steps*3)
            val[discrete] = 1
            discrete = val
        return read_supervised.dediscretize_all(discrete, self.conf.steering_steps, self.conf.INCLUDE_ACCPLUSBREAK)

    def discretize(self, throttle, brake, steer):
        return read_supervised.discretize_all(throttle, brake, steer, self.conf.steering_steps, self.conf.INCLUDE_ACCPLUSBREAK)

    def inflate_speed(self, speed):
        return read_supervised.inflate_speed(speed, self.conf.speed_neurons, self.conf.SPEED_AS_ONEHOT, 1) #is now normalized!
    
    def resetUnityAndServer(self):
        if self.containers.UnityConnected:
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
###################################################################################################  

  
class AbstractRLAgent(AbstractAgent):
    def __init__(self, conf, containers, isPretrain=False, start_fresh=False, *args, **kwargs):
        super().__init__(conf, containers, *args, **kwargs)
        self.start_fresh = start_fresh
        self.wallhitPunish = 1
        self.wrongDirPunish = 0 #10
        self.isPretrain = isPretrain
        self.show_plots = False  #wird im initForDriving ggf überschrieben
        self.startepsilon = self.conf.startepsilon #standard from config, but may be overridden
        self.minepsilon = self.conf.minepsilon #standard from config, but may be overridden
        self.finalepsilonframe = self.conf.finalepsilonframe #standard from config, but may be overridden
        self.steeraverage = deque(5*[0], 5)
        #memory will only be added in initfordriving
        

    ###########################################################################
    #################### functions that need to be implemented#################
    ###########################################################################
    
    #if not overridden, this can be used as a complete-random-agent
    def policyAction(self, agentState):
        return self.randomAction(agentState)

        
    ###########################################################################
    #################### functions that may be overwritten#####################        
    ###########################################################################        

    def initForDriving(self, *args, **kwargs):   
        assert not self.isPretrain, "You need to load the agent as Not-pretrain for a run!"
        assert self.containers is not None, "if you init the net for a run, the containers must not be None!"
        if not self.start_fresh:
            assert os.path.exists(self.folder(self.conf.pretrain_checkpoint_dir) or self.folder(self.conf.checkpoint_dir)), "I need any kind of pre-trained model"
        
        if not hasattr(self, "memory"): #einige agents haben bereits eine andere memory-implementation, die sollste nicht überschreiben
            self.memory = Memory(self.conf.memorysize, self.conf, self)  
        super().initForDriving(*args, **kwargs) 
        self.keep_memory = kwargs["keep_memory"] if "keep_memory" in kwargs else self.conf.keep_memory
        self.freezeInfReasons = []
        self.freezeLearnReasons = [] 
        self.numInferencesAfterLearn = 0
        self.numLearnAfterInference = 0
        self.epsilon = self.startepsilon
        #self.isinitialized = True  #muss jeder agent individuell am Ende machen!
    
        
    
    #hard rule for rewards: steering away from bad states cannot be better than being in a good state!
    def calculateReward(self, *gameState):
        vvec1_hist, vvec2_hist, otherinput_hist, action_hist = gameState
        self.steeraverage.append(action_hist[1][2]) 
        dist = otherinput_hist[0].CenterDist[0]-0.5  #abs davon ist 0 in der mitte, 0.15 vor dem curb, 0.25 mittig auf curb, 0.5 rand
        angle = otherinput_hist[0].SpeedSteer.carAngle - 0.5
               
        speed = otherinput_hist[0].SpeedSteer.speedInStreetDir*3 #Beim maxspeed von 80 maximal 1
        badspeed = abs(otherinput_hist[0].SpeedSteer.speedInTraverDir)*3
        
        stay_on_street = ((0.5-abs(dist))*2)+0.35 #jetzt ist größer 1 auf der street
        stay_on_street = stay_on_street**0.1 if stay_on_street > 1 else stay_on_street**2 #ON street not steep, OFF street very steep 
        stay_on_street = ((1-((0.5-abs(dist))*2))**10) * -self.wallhitPunish + (1-(1-((0.5-abs(dist))*2))**10) *  stay_on_street #the influence of wallhitpunish is exponentially more relevant the closer to the wall you are
        #in range [1,-1] for wallhitpunish=1
        prog = otherinput_hist[0].ProgressVec.Progress #"die dicken fische liegen hinten" <- extra reward for coming far
        prog = prog/10 if prog > 0 else 0 #becomes in range [0,0.1]
        direction_bonus = abs((0.5-(abs(angle)))*2/0.75) 
        direction_bonus = (direction_bonus**0.4 if direction_bonus > 1 else direction_bonus**2) / 1.1 / 2 #no big difference until 45degrees, then BIG diff.
        #maximally 0.5, minimally 0
        tmp = (np.mean(self.steeraverage))        
        steer_bonus1 = tmp/5 + angle #this one rewards sterering into street-direction if the cars angle is off...
        steer_bonus1 = 0 if np.sign(steer_bonus1) != np.sign(angle) and abs(angle) > 0.15 else steer_bonus1
        steer_bonus1 = (abs(dist*2)) * ((0.5-abs(angle)) * (1-abs(steer_bonus1))) + (1-abs(dist*2))*0.5  #more relevant the further off you are.
        steer_bonus2 = (1-((0.5-abs(dist))*2))**10 * -abs(((tmp+np.sign(dist))*np.sign(dist)))/1.5   #more relevant the furhter off, steering away from wall is as valuable as doing nothing in center, doing nothing is worse, steering towards sucks 
        #so steerbonus1+steerbonus2 is maximally 0.5
        
        #vor den kurven sind bestimmte werte irrelevanter
        curveMultiplier = 1-abs(otherinput_hist[0].SpeedSteer.CurvinessBeforeCar-0.5)
        direction_bonus *= curveMultiplier
        badspeed *= curveMultiplier 
        
        speed = speed-badspeed if speed-badspeed > 0 else 0
        
        
        rew = speed + 0.5 * stay_on_street + prog + 0.5 * direction_bonus + 0.5*(steer_bonus1+steer_bonus2)
        rew = max(rew, 0) #logik dahinter: wenn das auto neben der wand steht, dann entscheidet es sich doch bei sonst nur negativen rewards freiwillig dafür in die wand zu fahren um sein leiden zu beenden (-2 + 0*negativerwert größer -2+gamma*negativerwert)
        rew /= 2
        return rew


#    def calculateReward(self, *gameState):
#        vvec1_hist, vvec2_hist, otherinput_hist, action_hist = gameState
#        progress_old = otherinput_hist[3].ProgressVec.Progress
#        progress_new = otherinput_hist[0].ProgressVec.Progress
#        if progress_old > 90 and progress_new < 10:
#            progress_new += 100
#        progress = round(progress_new-progress_old,3)
#        stay_on_street = abs(otherinput_hist[0].CenterDist[0])
#        stay_on_street = round(0 if stay_on_street < 5 else self.wallhitPunish if stay_on_street >= 10 else stay_on_street/20, 3)
#        return progress-stay_on_street

        
    
    def randomAction(self, agentState):
        print("Random Action", level=2)
        action = np.random.randint(4) if self.conf.INCLUDE_ACCPLUSBREAK else np.random.randint(3)
        if action == 0: brake, throttle = 0, 1
        if action == 1: brake, throttle = 0, 0
        if action == 2: brake, throttle = 1, 0
        if action == 3: brake, throttle = 1, 1
        if agentState[2]: #"carstands"
            brake, throttle = 0, 1 
        #alternative 1a: steer = ((np.random.random()*2)-1)
        #alternative 1b: steer = min(max(np.random.normal(scale=0.5), 1), -1)
        #für 1a und 1b:  steer = read_supervised.dediscretize_steer(read_supervised.discretize_steering(steer, self.conf.steering_steps))
        #alternative 2:
        tmp = [0]*self.conf.steering_steps
        tmp[np.random.randint(self.conf.steering_steps)] = 1
        steer = read_supervised.dediscretize_steer(tmp)
        #throttle, brake, steer = 1, 0, 0
        result = "["+str(throttle)+", "+str(brake)+", "+str(steer)+"]"
        return result, (throttle, brake, steer)  #er returned immer toUse, toSave     
      

    ###########################################################################
    ########################### Necessary functions ###########################
    ###########################################################################


    #overridden from AbstractAgent
    #this assumes standard epsilon-greedy. If agent's differ from that, they can let randomaction return policyaction and use self.epsilon therein if needed
    def performAction(self, gameState, pastState):
        if self.isinitialized and self.checkIfAction():
            self.numsteps += 1
            self.repeated_action_for += 1
            self.addToMemory(gameState, pastState)
                
            if self.repeated_action_for < self.action_repeat:
                toUse, toSave = self.last_action 
            else:
                agentState = self.getAgentState(*gameState) #may be overridden
                if len(self.memory) >= self.conf.replaystartsize or self.epsilon == 0:
                    self.epsilon = min(round(max(self.startepsilon-((self.startepsilon-self.minepsilon)*((self.model.run_inferences()-self.conf.replaystartsize)/self.finalepsilonframe)), self.minepsilon), 5), 1)
                    if np.random.random() < self.epsilon:
                        toUse, toSave = self.randomAction(agentState)
                    else:
                        toUse, toSave = self.policyAction(agentState)
                else:
                    toUse, toSave = self.randomAction(agentState)
                self.last_action = toUse, toSave
              
#            print("####################")
#            vvec1_hist, vvec2_hist, otherinput_hist, action_hist = gameState
#            print(toSave)
#            print(otherinput_hist[0].SpeedSteer)
            
            self.containers.outputval.update(toUse, toSave, self.containers.inputval.CTimestamp, self.containers.inputval.STimestamp)  
            if self.conf.learnMode == "between":
                if self.numsteps % self.conf.ForEveryInf == 0 and self.canLearn():
                    print("freezing python because after", self.model.run_inferences(), "iterations I need to learn (between)", level=2)
                    self.freezeInf("LearningComes")
                    self.dauerLearnANN(self.conf.ComesALearn)
                    self.unFreezeInf("LearningComes")
                
            
        
    #gamestate and paststate sind jeweils (vvec1_hist, vvec2_hist, otherinputs_hist, action_hist) #TODO: nicht mit gamestate und paststate, direkt mit agentstate!
    def addToMemory(self, gameState, pastState): 
        assert hasattr(self, "memory") and self.memory is not None, "I don't have a memory, that's fatal."
        if type(pastState) in (np.ndarray, list, tuple): #nach reset/start ist pastState einfach False
            past_conv_inputs, past_other_inputs, _ = self.getAgentState(*pastState)
            s  = (past_conv_inputs, past_other_inputs)
            a  = self.getAction(*pastState)  #das (throttle, brake, steer)-tuple. 
            r = self.calculateReward(*gameState)
            conv_inputs, other_inputs, _ = self.getAgentState(*gameState)
            s2 = (conv_inputs, other_inputs)
            markovtuple = [s,a,r,s2,False] #not actually a tuple because punish&endepisode require something mutable
            self.memory.append(markovtuple)  
            print("adding to Memory:",a, r, level=4) 
            #values for evalation:
            stateval = self.model.statevalue(self.makeInferenceUsable(s))[0] #TODO: gucken ob stateval richtig ist!
            self.episode_statevals.append(stateval)
            return a, r, stateval, self.humantakingcontrolstring #damit agents das printen können wenn sie wollen
        return None, 0, 0, ""
      
       
     
    def checkIfAction(self):
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
                    return super().checkIfAction()
                self.numLearnAfterInference = 0
            self.numInferencesAfterLearn += 1
        #print(self.numLearnAfterInference, self.numInferencesAfterLearn, level=10)
        if self.model.run_inferences() >= self.conf.train_for: 
            return False
        else:
            return super().checkIfAction()

                
    def canLearn(self):
        return len(self.memory) > self.conf.batch_size+self.conf.history_frame_nr+1 and \
               len(self.memory) > self.conf.replaystartsize and self.model.run_inferences() < self.conf.train_for



    def dauerLearnANN(self, learnSteps):
        i = 0
        res = 0
        while self.containers.KeepRunning and self.model.run_inferences() <= self.conf.train_for and i < learnSteps:
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
                res += self.learnANN()
                if self.conf.ForEveryInf and self.conf.ComesALearn and self.conf.learnMode == "parallel":
                    self.numLearnAfterInference += 1
            i += 1         
        print(res/learnSteps)
        self.unFreezeInf("updateFrequency") #kann hier ruhig sein, da es eh nur unfreezed falls es aufgrund von diesem grund gefreezed war.
        if self.model.run_inferences() >= self.conf.train_for: #if you exited because you're completely done
            self.saveNet()
            print("Stopping learning because I'm done after", self.model.run_inferences(), "inferences", level=10)
        

            
    def learnANN(self):
        QLearnInputs = self.create_QLearnInputs_from_MemoryBatch(self.memory.sample(self.conf.batch_size))
        tmp = self.model.q_train_step(QLearnInputs, False)
        if self.model.step() > 0 and self.model.step() % self.conf.checkpointall == 0 or self.model.run_inferences() >= self.conf.train_for:
            self.saveNet()          
        return tmp

        
    def punishLastAction(self, howmuch):
        assert hasattr(self, "memory")
#        self.memory.punishLastAction(howmuch)
        print("punishLastAction removed")
        

    def endEpisode(self, reason, gameState):  #reasons are: turnedaround, timeover, resetserver, wallhit, rounddone
        assert hasattr(self, "memory")
        #TODO: die ersten 2 zeilen kann auch der abstractagent schon, dann muss ich im server nicht immer nach hasattr(memory) fragen!
        self.resetUnityAndServer()
        self.steeraverage = deque(5*[0], 5)
        self.episodes += 1        
        mem_epi_slice = self.memory.endEpisode() #bei actions, nach denen resettet wurde, soll er den folgestate nicht mehr beachten (später gucken wenn reset=true dann setze Q_DECAY auf quasi 100%)
        self.eval_episodeVals(mem_epi_slice, gameState, reason)

           
    def saveNet(self):
        self.freezeEverything("saveNet")
        self.model.save()
        if self.conf.save_memory_with_checkpoint and not self.model.isPretrain:
            self.memory.save_memory()
        self.unFreezeEverything("saveNet")

        
    def eval_episodeVals(self, mem_epi_slice, gameState, endReason):
        vvec1_hist, vvec2_hist, otherinput_hist, action_hist = gameState
        avg_rewards = round(self.memory.average_rewards(mem_epi_slice[0], mem_epi_slice[1]),3)
        avg_values = round(np.mean(np.array(self.episode_statevals)), 3)
        self.episode_statevals = []
        #other evaluation-values we need are time the agent took and percentage the agent made. However, becasue those values are not neccessarily
        #officially known to the agent (since agentstate != environmentstate), we need to take them from the environment-state
        progress = round(otherinput_hist[0].ProgressVec.Progress*100 if endReason != "lapdone" else 100, 2)
        laptime = round(otherinput_hist[0].ProgressVec.Laptime,1)
        valid = otherinput_hist[0].ProgressVec.fValidLap
        evalstring = "Avg-r:",avg_rewards,"Avg-Q:",avg_values,"progress:",progress,"laptime:",laptime,"(valid)" if valid else ""
        print(evalstring, level=8)
        self.evaluator.add_episode([avg_rewards, avg_values, progress, laptime], nr=self.episodes, startMemoryEntry=mem_epi_slice[0], endMemoryEntry=mem_epi_slice[1], endIteration=self.model.run_inferences(), reinfNetSteps=self.model.step(), endEpsilon=self.epsilon)
        return evalstring
        

    #needed in the pretrain-functions    
    def make_trainbatch(self,dataset,batchsize,epsilon=0):
        trainBatch = dataset.create_QLearnInputs_fromBatch(*dataset.next_batch(self.conf, self, batchsize), self)
        s,a,r,s2,t = trainBatch
        if np.random.random() < epsilon:
            r = np.zeros_like(r)
            a = np.array([random_unlike(i,self) for i in a])
            t = np.array([True]*len(t))
            trainBatch = s,a,r,s2,t
        return trainBatch


    def preTrain(self, dataset, iterations=None, supervised=False):    
        dataset.reset_batch()
        trainBatch = self.make_trainbatch(dataset,dataset.numsamples)
        print('Iteration %3d: Accuracy = %.2f%%' % (self.model.pretrain_episode(), self.model.getAccuracy(trainBatch, likeDDPG=False)), level=10)
        
        assert self.model.step() == 0, "I dont pretrain if the model already learned on real data!"
        iterations = self.conf.pretrain_iterations if iterations is None else iterations
        print("Starting pretraining", level=10)
        for i in range(iterations):
            start_time = time.time()
            self.model.inc_episode()
            dataset.reset_batch()
            while dataset.has_next(self.conf.pretrain_batch_size):
                trainBatch = self.make_trainbatch(dataset,self.conf.pretrain_batch_size,0.8)
                if supervised:
                    self.model.sv_train_step(trainBatch, True)
                else:
                    self.model.q_train_step(trainBatch, True)    
            if (i+1) % 25 == 0:
                self.model.save()    
            dataset.reset_batch()
            trainBatch = self.make_trainbatch(dataset,dataset.numsamples)
            print('Iteration %3d: Accuracy = %.2f%% (%.1f sec)' % (self.model.pretrain_episode(), self.model.getAccuracy(trainBatch, likeDDPG=False), time.time()-start_time), level=10)

    ###########################################################################
    ############################# Helper functions#############################
    ###########################################################################

    #memoryBatch is [[s,a,r,s2,t],[s,a,r,s2,t],[s,a,r,s2,t],...], what we want as Q-Learn-Input is [[s],[a],[r],[s2],[t]] 
    #.. to be more precise: [[(c,f),a,r,(c,f),t],[(c,f),a,r,(c,f),t],...]  and [[(c,f)],[a],[r],[(c,f)],[t]]
    def create_QLearnInputs_from_MemoryBatch(self, memoryBatch):
        visionroll = lambda vision: np.rollaxis(vision, 0, 3) if vision is not None else None 
        oldstates, actions, rewards, newstates, resetafters = zip(*memoryBatch)      
        #is already [[(c,f)],[a],[r],[(c,f)],[t]], however the actions are tuples, and we want argmax's... and netUsableOtherinputs
        actions = np.array([self.makeNetUsableAction((throttle, brake, steer)) for throttle, brake, steer in actions]) 
        oldstates = [(visionroll(i[0]), np.array(self.makeNetUsableOtherInputs(i[1]))) for i in oldstates]
        newstates = [(visionroll(i[0]), np.array(self.makeNetUsableOtherInputs(i[1]))) for i in newstates]#
        return oldstates, actions, np.array(rewards), newstates, np.array(resetafters)
        
        
    def freezeEverything(self, reason):
        self.freezeLearn(reason)
        self.freezeInf(reason)

    def freezeLearn(self, reason):
        if not reason in self.freezeLearnReasons:
            self.containers.freezeLearn = True
            self.freezeLearnReasons.append(reason)

    def freezeInf(self, reason):
        if self.containers.UnityConnected:
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
        if self.containers.UnityConnected:
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
