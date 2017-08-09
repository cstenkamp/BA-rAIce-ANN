# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 22:09:03 2017

@author: csten_000
"""

import sys
import os
import numpy as np

class Config():
    #FOLDER STUFF
    LapFolderName = "SavedLaps/"
    pretrain_log_dir = "PT_SummaryLogs/"  
    pretrain_checkpoint_dir = "PT_Checkpoints/"
    log_dir = "RL_SummaryLogs/"  
    checkpoint_dir = "RL_Checkpoints/"
    memory_dir = "./" #will be a pickle-file    
    xml_dir = "./"
    save_xml = True
    #wir haben super-über-ordner für RLLearn, checkpoint, summarylogdir & memory für jede kombi aus hframes, secondcam, mspersec... und dann der agent-folder
    
    #PRETRAIN STUFF
    pretrain_batch_size = 32
    pretrain_iterations = 90     #90, 120
    pretrain_lrdecayafter = pretrain_iterations//2  #//3 für 90, 120
    pretrain_checkpointall = 10
    pretrain_summaryall = False
    pretrain_keep_prob = 0.8
    pretrain_initscale = 0.1
    
    pretrain_sv_initial_lr = 0.005
    pretrain_sv_lr_decay = 0.9
    pretrain_sv_minimal_lr = 1e-6 
    
    pretrain_q_initial_lr = 0.00005
    pretrain_q_lr_decay = 0.999
    pretrain_q_minimal_lr = 0.000001
    
    #REINF_LEARN STUFF
    train_for = sys.maxsize-1
    initial_lr = 0.00001  #0.00005 
    lr_decay = 0.9999 #hier gehs um steps, nicht mehr um episoden!
    lrdecayafter = train_for//2
    minimal_lr = 0.000001
    
    target_update_tau = 0.001 
    batch_size = 32
    startepsilon = 0.2
    finalepsilonframe = 100000  #I could use epsilondecrease = 0.0001 instead, however then every new run epsilon would reset to startepsilon (since it doesn't depend on numIterations then)
    minepsilon = 0.005
    q_decay = 0.99
    checkpointall = 1000 #RLsteps, not inferences!
    
    #LEARN SETTINGS    
    ForEveryInf, ComesALearn = 400, 100
    wallhit_ends_episode = True #problem: wenn es das nicht tun würde, würde er beim lernen den Q-val des folgestates mitbeachten und denken "oh, gegen die Wand fahren ist voll gut"...
    lapdone_ends_episode = True
    time_ends_episode = 60 #sekunden oder False
    save_memory_with_checkpoint = True
    save_memory_on_exit = False
    save_memory_all_mins = False
    replaystartsize = 40000
    memorysize = 50000
    use_constantbutbigmemory = False
    visionvecdtype = np.int8 #wäre es np.bool würde er den rand als street sehen!
    keep_memory = True
    learnMode = "between" #"parallel", "between", "remote" (the latter is tobedone)

    
    #GAME SETTINGS
    history_frame_nr = 4 #incl. dem jetzigem!
    speed_neurons = 30 
    SPEED_AS_ONEHOT = False    
    image_dims = [30,45] 
    msperframe = 100 #50   #ACHTUNG!!! Dieser wert wird von unity überschrieben!!!!! #TODO: dass soll mit unity abgeglichen werden!
    use_cameras = True
    use_second_camera = True
    MAXSPEED = 250
    #for discretized algorithms
    steering_steps = 7
    INCLUDE_ACCPLUSBREAK = False
    #for continuus algorithms
    num_actions = 3
    action_bounds = [(0, 1), (0, 1), (-1, 1)]
    actor_lr = 0.0001
    critic_lr = 0.001
    
    #DEBUG STUFF
    UPDATE_ONLY_IF_NEW = False #sendet immer nach jedem update -> Wenn False sendet er wann immer er was kriegt
    reset_if_wrongdirection = True

    
    def has_gpu(self):
        from tensorflow.python.client import device_lib
        return "gpu" in ",".join([x.name for x in device_lib.list_local_devices()])
    
    def superfolder(self):
        numcams = "0cams_" if not self.use_cameras else ("2cams_" if self.use_second_camera else "1cam_")
        return "data/data_"+str(self.history_frame_nr)+"hframes_"+numcams+ str(self.msperframe) + "msperframe/"

    def __init__(self):
        assert not (self.use_second_camera and (self.history_frame_nr == 1)), "If you're using 2 cameras, you have to use historyframes!"
        assert os.path.exists(self.LapFolderName), "No data to train on at all!"        
        if self.learnMode == "parallel" and not self.has_gpu(): self.learnMode = "between"



    
class DQN_Config(Config):
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
        
        
    
    
    
class Half_DQN_Config(Config):
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
    
