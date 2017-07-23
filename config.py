# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 22:09:03 2017

@author: csten_000
"""

import sys
import os
import numpy as np

class Config(object):
    LapFolderName = "SavedLaps/"
    log_dir = "SV_SummaryLogs/"  
    checkpoint_dir = "SV_Checkpoints/"
    #wir haben super-über-ordner für RLLearn, checkpoint, summarylogdir & memory für jede kombi aus hframes, secondcam, mspersec
    
    history_frame_nr = 4 #incl. dem jetzigem!
    speed_neurons = 30 #wenn null nutzt er sie nicht
    SPEED_AS_ONEHOT = False
    #for discretized algorithms
    steering_steps = 7
    INCLUDE_ACCPLUSBREAK = False
    #for continuus algorithms
    num_actions = 3
    UPDATE_ONLY_IF_NEW = False #sendet immer nach jedem update -> Wenn False sendet er wann immer er was kriegt
    
    
    reset_if_wrongdirection = True
    
    image_dims = [30,45] 
    msperframe = 100 #50   #ACHTUNG!!! Dieser wert wird von unity überschrieben!!!!! #TODO: dass soll mit unity abgeglichen werden!
    use_second_camera = True
    
    batch_size = 32
    keep_prob = 0.8
    initscale = 0.1
    max_grad_norm = 10
    
    iterations = 90     #90, 120
    initial_lr = 0.005
    lr_decay = 0.9
    lrdecayafter = iterations//2  #//3 für 90, 120
    minimal_lr = 1e-6 #mit diesen settings kommt er auf 0.01 loss, 99.7% correct inferences
    checkpointall = 10
    
    def __init__(self):
        assert not (self.use_second_camera and (self.history_frame_nr == 1)), "If you're using 2 cameras, you have to use historyframes!"
        assert os.path.exists(self.LapFolderName), "No data to train on at all!"        
        
        self.log_dir = self.superfolder()+self.log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)         
            
        self.checkpoint_dir = self.superfolder()+self.checkpoint_dir
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir) 

    def superfolder(self):
        return "data/data_"+str(self.history_frame_nr)+"hframes_"+("2cams_" if self.use_second_camera else "1cam_") + str(self.msperframe) + "msperframe/"




class RL_Config(Config):
    log_dir = "RL_SummaryLogs/"  
    checkpoint_dir = "RL_Checkpoints/"
    savememorypath = "./" #will be a pickle-file
     
    keep_prob = 1
    max_grad_norm = 10
    initial_lr = 0.001
    #lr_decay = 1
    
    startepsilon = 0.2
    epsilondecrease = 0.0001
    minepsilon = 0.005
    batchsize = 32
    q_decay = 0.99
    checkpointall = 300 #RLsteps, not inferences!
    copy_target_all = 100
    
    replaystartsize = 0
    memorysize = 30000
    use_efficientmemory = True
    use_constantbutbigmemory = False
    visionvecdtype = np.int8 #wäre es np.bool würde er den rand als street sehen!
    keep_memory = True
    saveMemoryAllMins = 45
    train_for = sys.maxsize-1
       
    ForEveryInf, ComesALearn = 40, 10
    learnMode = "between" #"parallel", "between", "remote" (the latter is tobedone)
   
    #re-uses history_frame_nr, image_dims, steering_steps, speed_neurons, INCLUDE_ACCPLUSBREAK, SPEED_AS_ONEHOT
    
    def device_has_gpu(self):
        from tensorflow.python.client import device_lib
        return "gpu" in ",".join([x.name for x in device_lib.list_local_devices()])

    def __init__(self):     
        self.savememorypath = self.superfolder()+self.savememorypath
        if not os.path.exists(self.savememorypath):
            os.makedirs(self.savememorypath)
                                              
        self.log_dir = self.superfolder()+self.log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)     
            
        self.checkpoint_dir = self.superfolder()+self.checkpoint_dir 
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)                 
                                    
        assert os.path.exists(Config().checkpoint_dir), "I need a pre-trained model"

        if self.learnMode == "parallel" and not self.device_has_gpu(): self.learnMode = "between"


    
class DQN_Config(RL_Config):
    batch_size = 32                 #minibatch size
    memorysize = 1000000            #replay memory size
    history_frame_nr = 4            #agent history length
    copy_target_all = 10000         #target network update frequency (C)
    q_decay = 0.99                  #discount factor
    #action repeat & noop-max
    initial_lr = 0.00025            #learning rate used by RMSProp
    lr_decay = 1                    #as the lr seems to stay equal, no decay
    rms_momentum = 0.95             #gradient momentum (=squared gradient momentum)
    min_sq_grad = 0.1               #min squared gradient 
    startepsilon = 1                #initial exploration
    minepsilon = 0.1                #final exploration
    finalepsilonframe = 1000000     #final exploration frame
    replaystartsize = 50000         #replay start size
    train_for = 50000000            #number of iterations to train for 
    ForEveryInf, ComesALearn = 4, 1 #update frequency & how often it checks it
    use_constantbutbigmemory = True
    keep_memory = True

    def __init__(self):
        super().__init__()
        
        
    
class Half_DQN_Config(RL_Config):
    batch_size = 32                     #minibatch size
    memorysize = 100000                 #replay memory size
    history_frame_nr = 4                #agent history length
    copy_target_all = 2000              #target network update frequency (C)
    q_decay = 0.99                      #discount factor
    startepsilon = 1                    #initial exploration
    minepsilon = 0.01                   #final exploration
    finalepsilonframe = 200000          #final exploration frame
    replaystartsize = 2000              #replay start size
    train_for = 30000000                #number of iterations to train for 
    ForEveryInf, ComesALearn = 400, 100 #update frequency & how often it checks it
    use_constantbutbigmemory = True
    keep_memory = True
    
    def __init__(self):
        super().__init__()
    
