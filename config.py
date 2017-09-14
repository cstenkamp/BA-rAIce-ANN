# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 22:09:03 2017

@author: csten_000
"""

import sys
import os
import numpy as np
np.set_printoptions(suppress=True)

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
    agents_file_dir = "./agents/"
    models_file_dir = "./models/"
    standardAgent = "ddpg_novision_rl_agent"
    standardSVAgent = "dqn_sv_agent"
    
    #PRETRAIN STUFF
    pretrain_batch_size = 128
    pretrain_iterations = 200
    pretrain_lrdecayafter = pretrain_iterations//2  #//3 für 90, 120
    pretrain_checkpointall = 10
    pretrain_summaryall = False
    pretrain_keep_prob = 0.8
    pretrain_initscale = 0.1
    pretrain_sv_initial_lr = 0.005
    pretrain_sv_lr_decay = 0.9
    pretrain_sv_minimal_lr = 1e-6 
    pretrain_q_initial_lr = 0.0025 #0.00025
    pretrain_q_lr_decay = 0.999
    pretrain_q_minimal_lr = 0.000001
    
    #REINF_LEARN STUFF
    train_for = sys.maxsize-1
    initial_lr = 0.000025  #0.000025
    lr_decay = 0.9999 #hier gehs um steps, nicht mehr um episoden!
    lrdecayafter = train_for//2
    minimal_lr = 0.000001
    target_update_tau = 0.001 #0.001
    batch_size = 64
    startepsilon = 0.25
    minepsilon = 0.001
    finalepsilonframe = 500000
    q_decay = 0.99
    checkpointall = 2000 #RLsteps, not inferences!
    summarize_tensorboard_allstep = 500 #false or number
    
    #LEARN SETTINGS    
    ForEveryInf, ComesALearn = 400, 100
    save_memory_with_checkpoint = True
    save_memory_on_exit = False
    save_memory_all_mins = False
    replaystartsize = 40 #4000
    memorysize = 500000
    use_constantbutbigmemory = False
    visionvecdtype = np.int8 #wäre es np.bool würde er den rand als street sehen!
    keep_memory = True
    learnMode = "between" #"parallel", "between", "remote" (the latter is tobedone)
    
    
    #GAME SETTINGS
    history_frame_nr = 4 #incl. dem jetzigem!
    action_repeat = 2
    headstart_num = 5
    speed_neurons = 20 
    SPEED_AS_ONEHOT = False    
    image_dims = [30,45] 
    msperframe = 100 #50   #ACHTUNG!!! Dieser wert wird von unity überschrieben!!!!! #TODO: dass soll mit unity abgeglichen werden!
    use_cameras = True
    use_second_camera = True
    MAXSPEED = 250
    INCLUDE_ACCPLUSBREAK = False
    
    #for discretized algorithms
    steering_steps = 7
    #for continuus algorithms
    num_actions = 3
    action_bounds = [(0, 1), (0, 1), (-1, 1)]
    throttle_index = 0
    brake_index = 1
    steer_index = 2
    actor_lr = 0.0001
    critic_lr = 0.001
    
    #DEBUG STUFF
    UPDATE_ONLY_IF_NEW = False #sendet immer nach jedem update -> Wenn False sendet er wann immer er was kriegt
    showColorArea = False
    use_settozero = True
    
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
        sys.path.append(self.agents_file_dir)
        sys.path.append(self.models_file_dir)
        self.dnum_actions = self.steering_steps*4 if self.INCLUDE_ACCPLUSBREAK else self.steering_steps*3 
        self.conv_stacksize = (self.history_frame_nr*2 if self.use_second_camera else self.history_frame_nr) 
        self.ff_stacksize = self.history_frame_nr 


    
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

    def __init__(self):
        super().__init__()
        
