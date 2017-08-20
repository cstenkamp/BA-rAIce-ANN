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


def main(conf, agentname, containers, start_fresh):

    agentclass = __import__(agentname).Agent
    myAgent = agentclass(conf, containers, isPretrain=True, start_fresh=start_fresh)    
    
    tf.reset_default_graph()                                                          
    trackingpoints = read_supervised.TPList(conf.LapFolderName, conf.use_second_camera, conf.msperframe, conf.steering_steps, conf.INCLUDE_ACCPLUSBREAK)
    print("Number of samples:",trackingpoints.numsamples)
    myAgent.preTrain(trackingpoints, 200)
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
            
    main(conf, agentname, containers, ("-startfresh" in sys.argv))