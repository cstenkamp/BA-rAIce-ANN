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
####own classes###
from myprint import myprint as print
import read_supervised
import infoscreen
import dqn
from evaluator import evaluator
from inefficientmemory import Memory

###################################################################################################

class AbstractAgent(object):
    def __init__(self, sv_conf, containers, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lock = threading.Lock()
        self.isinitialized = False
        self.containers = containers  
        self.sv_conf = sv_conf
        self.numIterations = 0
        self.ff_inputsize = 0
        self.usesConv = True
        self.conv_stacked = True
        self.ff_stacked = False
        self.network = None
        self.graph = tf.Graph()
        self.network = dqn.CNN
        
    ##############functions that should be impemented##########
    def checkIfInference(self):
        if self.containers.sv_conf.UPDATE_ONLY_IF_NEW and self.containers.inputval.alreadyread:
            return False
        return True
        
    def preRunInference(self):       
        self.numIterations += 1
    
    def postRunInference(self, toUse, toSave): #toUse will already be a prepared string, toSave will be raw.
        self.containers.outputval.update(toUse, toSave, self.containers.inputval.CTimestamp, self.containers.inputval.STimestamp)  
    
    #creates the Agents state from the real state. this is the base version, other agents may overwrite it.
    def getAgentState(self, vvec1_hist, vvec2_hist, otherinput_hist, action_hist): 
        assert self.sv_conf.use_cameras, "You disabled cameras in the config, which is impossible for this agent!"
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
        #TODO: wenn du die action als 3 werte speichest, dann...
        return self.discretize(*action)
        
    def performNetwork(self, _, __, ___):
        print("Another ANN Inference", level=3)
        
        
    def initNetwork(self, start_fresh):
        raise NotImplementedError    


    def svTrain(self):
        with self.graph.as_default(): 
            trackingpoints = read_supervised.TPList(self.sv_conf.LapFolderName, self.sv_conf.use_second_camera, self.sv_conf.msperframe, self.sv_conf.steering_steps, self.sv_conf.INCLUDE_ACCPLUSBREAK)
            print("Number of samples: %s | Tracking every %s ms with %s historyframes" % (trackingpoints.numsamples, str(self.sv_conf.msperframe), str(self.sv_conf.history_frame_nr)), level=10)

            initializer = tf.random_uniform_initializer(-0.1, 0.1) #bei variablescopes kann ich nen default-initializer für get_variables festlegen
                              
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                sv_model = self.network(self.sv_conf, self, mode="sv_train")
              
            init = tf.global_variables_initializer()
            sv_model.trainvars["global_step"] = sv_model.global_step #TODO: try to remove this and see if it still works, cause it should
            saver = tf.train.Saver(sv_model.trainvars, max_to_keep=2)
    
            with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                summary_writer = tf.summary.FileWriter(self.folder(self.sv_conf.log_dir), sess.graph) #aus dem toy-example
                
                sess.run(init)
                ckpt = tf.train.get_checkpoint_state(self.folder(self.sv_conf.checkpoint_dir))
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    stepsPerIt = trackingpoints.numsamples//self.sv_conf.batch_size
                    already_run_iterations = sv_model.global_step.eval()//stepsPerIt
                    sv_model.iterations = already_run_iterations
                    print("Restored checkpoint with",already_run_iterations,"Iterations run already", level=10) 
                else:
                    already_run_iterations = 0
                    
                num_iterations = self.sv_conf.iterations - already_run_iterations
                print("Running for",num_iterations,"further iterations" if already_run_iterations>0 else "iterations", level=10)
                for _ in range(num_iterations):
                    start_time = time.time()
    
                    step = sv_model.global_step.eval() 
                    train_loss = sv_model.run_sv_train_epoch(sess, trackingpoints, summary_writer)
                    
                    savedpoint = ""
                    if sv_model.iterations % self.sv_conf.checkpointall == 0 or sv_model.iterations == self.sv_conf.iterations:
                        checkpoint_file = os.path.join(self.folder(self.sv_conf.checkpoint_dir), 'model.ckpt')
                        saver.save(sess, checkpoint_file, global_step=step)       
                        savedpoint = "(checkpoint saved)"
                    
                    print('Iteration %3d (step %4d): loss = %.2f (%.3f sec)' % (sv_model.iterations, step+1, train_loss, time.time()-start_time), savedpoint, level=8)
                    
                    
                ev, loss, _ = sv_model.run_sv_eval(sess, trackingpoints)
                print("Result of evaluation:", level=8)
                print("Loss: %.2f,  Correct inferences: %.2f%%" % (loss, ev*100), level=10)
    
    #            dataset.reset_batch()
    #            _, visionvec, _, _ = dataset.next_batch(config, 1)
    #            visionvec = np.array(visionvec[0])
    #            print(cnn.run_inference(sess,visionvec, self.stacksize))
 
    
    #################### Helper functions#######################
    def dediscretize(self, discrete):
        return read_supervised.dediscretize_all(discrete, self.sv_conf.steering_steps, self.sv_conf.INCLUDE_ACCPLUSBREAK)

    def discretize(self, throttle, brake, steer):
        return read_supervised.discretize_all(throttle, brake, steer, self.sv_conf.steering_steps, self.sv_conf.INCLUDE_ACCPLUSBREAK)

    def inflate_speed(self, speed):
        return read_supervised.inflate_speed(speed, self.sv_conf.speed_neurons, self.sv_conf.SPEED_AS_ONEHOT, self.sv_conf.MAXSPEED)
    
    def resetUnityAndServer(self):
        import server
        server.resetUnityAndServer(self.containers)

    def folder(self, actualfolder):
        folder = self.sv_conf.superfolder()+self.name+"/"+actualfolder
        if not os.path.exists(folder):
            os.makedirs(folder)   
        return folder

###################################################################################################   
###################################################################################################   
###################################################################################################   
###################################################################################################  

  
class AbstractRLAgent(AbstractAgent):
    def __init__(self, sv_conf, containers, rl_conf, *args, **kwargs):
        super().__init__(sv_conf, containers, *args, **kwargs)
        self.rl_conf = rl_conf
        self.last_random_timestamp = 0
        self.last_random_action = None
        self.repeat_random_action_for = 1000
        if not hasattr(self, "memory"): #einige agents haben bereits eine andere memory-implmentation, die sollste nicht überschreiben
            self.memory = Memory(rl_conf.memorysize, containers, self)
        self.reinfNetSteps = 0
        self.numInferencesAfterLearn = 0
        self.numLearnAfterInference = 0
        self.freezeInfReasons = []
        self.freezeLearnReasons = [] 
        self.wallhitPunish = 1;
        self.wrongDirPunish = 10;
        self.episode_statevals = [] 
        self.episodes = 0
        self.evaluator = evaluator(self.containers, self, self.containers.show_plots, self.containers.sv_conf.save_xml,      \
                                   ["average rewards", "average Q-vals",      "progress", "laptime"                       ], \
                                   [1,                 self.sv_conf.MAXSPEED, 100,         self.rl_conf.time_ends_episode ] )

    def addToMemory(self, gameState, pastState): 
        assert self.memory is not None, "It should be specified in server right afterwards"
        
        if pastState[0] is not None: #was der Fall DIREKT nach reset oder nach start ist
            
            past_conv_inputs, past_other_inputs, _ = self.getAgentState(*pastState)
            s  = (past_conv_inputs, past_other_inputs)
            a  = self.getAction(*pastState)
            r = self.calculateReward(*gameState)
            conv_inputs, other_inputs, _ = self.getAgentState(*gameState)
            s2 = (conv_inputs, other_inputs)
        
            if not self.SAVE_ACTION_AS_ARGMAX: #action ist entweder das argmax der final_neurons ODER das (throttle, brake, steer)-tuple
                actuAction = a                                                                                         
                a = self.memory.make_long_from_floats(*a)
            else:
                actuAction = self.dediscretize(a, self.rl_conf)
            
            self.memory.append([s, a, r, s2, False])  
            
            print("adding to Memory:",actuAction, r, level=4) 
            
            #values for evalation:
            stateval = self.target_cnn.calculate_value(self.session, conv_inputs, self.makeNetUsableOtherInputs(other_inputs))[0]
            self.episode_statevals.append(stateval)
            
            
            
            if self.containers.showscreen:
                conv_inputs, other_inputs, _ = self.getAgentState(*gameState)
                infoscreen.print(actuAction, round(r,2), round(stateval,2), self.humantakingcontrolstring, containers= self.containers, wname="Last memory")
                if len(self.memory) % 20 == 0:
                    infoscreen.print(">"+str(len(self.memory)), containers= self.containers, wname="Memorysize")
      
       
     
    def checkIfInference(self):
        if self.containers.freezeInf:
            return False
        #hier gehts darum die Inference zu freezen bis das learnen eingeholt hat. (falls update_frequency gesetzt)
        if self.rl_conf.ForEveryInf and self.rl_conf.ComesALearn and self.canLearn() and self.rl_conf.learnMode == "parallel":
            if self.numLearnAfterInference == self.rl_conf.ComesALearn and self.numInferencesAfterLearn == self.rl_conf.ForEveryInf:
                self.numLearnAfterInference = self.numInferencesAfterLearn = 0            
                self.unFreezeLearn("updateFrequency")   
                self.unFreezeInf("updateFrequency")
             #Alle ForEveryInf inferences sollst du warten, bis ComesALearn mal in der zwischenzeit gelernt wurde.
            if self.numInferencesAfterLearn == self.rl_conf.ForEveryInf:
                #gucke ob er in der zwischenzeit ComesALearn mal gelernt hat, wenn nein, freeze Inference
                self.unFreezeLearn("updateFrequency")      
                if self.numLearnAfterInference < self.rl_conf.ComesALearn:
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
        #ACHTUNG! self.containers.inputval.addResultAndBackup(toSave)  WURDE HIER ENTFERNT!
        if self.containers.rl_conf.learnMode == "between":
            print("freezing python because after", self.numIterations, "iterations I need to learn (between)", level=2)
            if self.numIterations % self.containers.rl_conf.ForEveryInf == 0:
                self.freezeInf("LearningComes")
                self.dauerLearnANN(self.containers.rl_conf.ComesALearn)
                self.unFreezeInf("LearningComes")


    def canLearn(self):
        return len(self.memory) > self.rl_conf.batchsize+self.rl_conf.history_frame_nr+1 and \
               len(self.memory) > self.rl_conf.replaystartsize and self.numIterations < self.rl_conf.train_for



    def dauerLearnANN(self, learnSteps):

        i = 0
        while self.containers.KeepRunning and self.numIterations <= self.rl_conf.train_for and i < learnSteps:
            cando = True
            #hier gehts darum das learnen zu freezen bis die Inference eingeholt hat. (falls update_frequency gesetzt)
            if self.rl_conf.ForEveryInf and self.rl_conf.ComesALearn and self.rl_conf.learnMode == "parallel":
                if self.numLearnAfterInference == self.rl_conf.ComesALearn and self.numInferencesAfterLearn == self.rl_conf.ForEveryInf:
                    self.numLearnAfterInference = self.numInferencesAfterLearn = 0
                    self.unFreezeLearn("updateFrequency")   
                    self.unFreezeInf("updateFrequency")
                #Alle ComesALearn sollst du warten, bis ForEveryInf mal zwischenzeitlich Inference gemacht wurde
                if self.numLearnAfterInference >= self.rl_conf.ComesALearn:
                    self.unFreezeInf("updateFrequency") 
                    if self.numInferencesAfterLearn < self.rl_conf.ForEveryInf and self.canLearn():
                        self.freezeLearn("updateFrequency")
                        print("FREEZELEARN", self.numLearnAfterInference, self.numInferencesAfterLearn, level=2)
                        cando = False
                    else:
                        self.numInferencesAfterLearn = 0
            if cando and not self.containers.freezeLearn and self.canLearn():
                self.learnANN()
                if self.rl_conf.ForEveryInf and self.rl_conf.ComesALearn and self.rl_conf.learnMode == "parallel":
                    self.numLearnAfterInference += 1
            i += 1
                                         
        self.unFreezeInf("updateFrequency")
        if self.numIterations >= self.rl_conf.train_for: #if you exited because you're completely done
            self.saveNet()
            print("Stopping learning because I'm done after", self.numIterations, "inferences", level=10)
        
        

    def saveNet(self):
        raise NotImplementedError
                
    def learnANN(self):
        raise NotImplementedError



        
#    def calculateReward(self, inputval):
#        progress_old = inputval.previous_otherinputs.ProgressVec.Progress
#        progress_new = inputval.otherinputs.ProgressVec.Progress
#        if progress_old > 90 and progress_new < 10:
#            progress_new += 100
#        progress = round(progress_new-progress_old,3)*200
#        
#        stay_on_street = abs(inputval.otherinputs.CenterDist)
#        #wenn er >= 10 war und seitdem keine neue action kam, muss er >= 10 bleiben!
#        
#        stay_on_street = round(0 if stay_on_street < 5 else self.wallhitPunish if stay_on_street >= 10 else stay_on_street-5, 3)
#        
#        return progress-stay_on_street


    def calculateReward(self, vvec1_hist, vvec2_hist, otherinput_hist, action_hist):
        stay_on_street = abs(otherinput_hist[0].CenterDist)
        stay_on_street = round(0 if stay_on_street < 5 else self.wallhitPunish if stay_on_street >= 10 else stay_on_street/10, 3)
        speed = otherinput_hist[0].SpeedSteer.speedInStreetDir / self.sv_conf.MAXSPEED
        return speed - stay_on_street


    
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
            #für 1a und 1b:  steer = read_supervised.dediscretize_steer(read_supervised.discretize_steering(steer, self.rl_conf.steering_steps))
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
        
        if self.SAVE_ACTION_AS_ARGMAX:
            return result, self.discretize(throttle, brake, steer, config) #er returned immer toUse, toSave
        else:
            return result, (throttle, brake, steer)                        #er returned immer toUse, toSave     



    def endEpisode(self, reason, gameState):  #reasons are: turnedaround, timeover, resetserver, wallhit, rounddone
        self.resetUnityAndServer()
        self.episodes += 1        
        if self.containers.usememory: #bei actions, nach denen resettet wurde, soll er den folgestate nicht mehr beachten (später gucken wenn reset=true dann setze Q_DECAY auf quasi 100%)
            episode = self.memory.endEpisode()
            self.print_episodeVals(episode, gameState, reason)



    def print_episodeVals(self, mem_epi_slice, gameState, endReason):
        avg_rewards = round(self.memory.average_rewards(slice(*mem_epi_slice)),3)
        avg_values = round(np.mean(np.array(self.episode_statevals)), 3)
        self.episode_statevals = []
        #other evaluation-values we need are time the agent took and percentage the agent made. However, becasue those values are not neccessarily
        #officially known to the agent (since agentstate != environmentstate), we need to take them from the environment-state
        progress = round(gameState[2][0].ProgressVec.Progress if endReason != "lapdone" else 100, 2)
        laptime = round(gameState[2][0].ProgressVec.Laptime,1)
        valid = gameState[2][0].ProgressVec.fValidLap
        print("Avg-r:",avg_rewards,"Avg-Q:",avg_values,"progress:",progress,"laptime:",laptime,"(valid)" if valid else "", level=8)
        if self.containers.showscreen:
                infoscreen.print("rw:", avg_rewards, "Q:", avg_values, "prg:", progress, "time:", laptime, "(v)" if valid else "", containers=self.containers, wname="Last Epsd")
        
        self.evaluator.add_episode(mem_epi_slice[0], mem_epi_slice[1], self.episodes, self.numIterations, self.reinfNetSteps, [avg_rewards, avg_values, progress, laptime])
        
                         
            
            
    def punishLastAction(self, howmuch):
        if self.containers.usememory:
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



