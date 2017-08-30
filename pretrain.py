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


def main(conf, agentname, containers, start_fresh, fake_real, numiters, supervised=False):

    agentclass = __import__(agentname).Agent
    myAgent = agentclass(conf, containers, isPretrain=(not fake_real), start_fresh=start_fresh)    
    
    tf.reset_default_graph()                                                          
    
    if not fake_real:
        trackingpoints = read_supervised.TPList(conf.LapFolderName, conf.use_second_camera, conf.msperframe, conf.steering_steps, conf.INCLUDE_ACCPLUSBREAK)
        print("Number of samples:",trackingpoints.numsamples)   
        assert trackingpoints.numsamples > 0, "You have no pre-training data!"
        if supervised:
            print("Training supervisedly! Not recommended!")
            myAgent.preTrain(trackingpoints, numiters, supervised=True)
        else:
            myAgent.preTrain(trackingpoints, numiters) #dann nimm die standard-einstellung! nicht epxlizit false
    else:
        assert not supervised, "Supervised fake_real training is not possible"
        itperlearn = myAgent.conf.ForEveryInf / myAgent.conf.ComesALearn
        iterations = int(myAgent.conf.train_for//itperlearn if numiters is None else numiters)
        myAgent.initForDriving(keep_memory=False, show_plots=False, use_evaluator=False)
        for i in range(iterations):
            maxQ = myAgent.learnANN() 
            if i % 250 == 0:
                print("Iteration",i,"max-Q-Val:",maxQ)
    




if __name__ == '__main__':  
    conf = config.Config()
    containers = Containers()

    if "--iterations" in sys.argv:
        num = sys.argv.index("--iterations")
        try:
            numiters = sys.argv[num+1]
            if numiters[0] == "-": raise IndexError
        except IndexError:
            print("With the '--iterations'-Parameter, you need to specify the number of iteratios!!")
            exit(0)
        numiters = int(numiters)
    else:
        numiters = None
        
        

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
            
    main(conf, agentname, containers, ("-startfresh" in sys.argv), ("-fakereal" in sys.argv), numiters, ("-supervised" in sys.argv))