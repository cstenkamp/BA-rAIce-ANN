# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 21:04:30 2017

@author: csten_000
"""

import tensorflow as tf
import time
#====own classes====
from myprint import myprint as print
import sys
import config
import read_supervised
from server import Containers


def main(conf, agentname, containers, start_fresh, fake_real):

    agentclass = __import__(agentname).Agent
    myAgent = agentclass(conf, containers, isPretrain=(not fake_real), start_fresh=start_fresh)    
        
    #TODO: das fake_real muss noch so geändert werden dass es ZWEI optionen gibt: fake_real UND dass man AUSSUCHEN KANN ob von memory oder von pretrain!!!!
    
    tf.reset_default_graph()                                                          
    trackingpoints = read_supervised.TPList(conf.LapFolderName, conf.use_second_camera, conf.msperframe, conf.steering_steps, conf.INCLUDE_ACCPLUSBREAK)
    print("Number of samples:",trackingpoints.numsamples)
    
    if not fake_real:
        myAgent.preTrain(trackingpoints, 200)
    else:
        myAgent.initForDriving()
        for i in range(20):
            for i in range(200):
                trainBatch = myAgent.create_QLearnInputs_from_MemoryBatch(myAgent.memory.sample(conf.batch_size))
                myAgent.model.q_learn(trainBatch, False)
            myAgent.model.save()        
    
    time.sleep(999)




if __name__ == '__main__':  
    conf = config.Config()
    containers = Containers()


    if "--agent" in sys.argv:
        num = sys.argv.index("--agent")
        try:
            agentname = sys.argv[num+1]
            if agentname[0] == "-": raise IndexError
        except IndexError:
            print("With the '--agent'-Parameter, you need to specify an agent!")
            exit(0)
    else:
        if "-svplay" in sys.argv:
            agentname = "dqn_sv_agent"
        else:
            agentname = "dqn_rl_agent"
            
    main(conf, agentname, containers, ("-startfresh" in sys.argv), ("-fakereal" in sys.argv))