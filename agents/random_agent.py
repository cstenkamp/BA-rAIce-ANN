import numpy as np
#====own classes====
from agent import AbstractAgent

class Agent(AbstractAgent):    
    def init(self, conf, containers, isPretrain=False, start_fresh=False, *args, **kwargs):
        self.name = "random_agent"
        super().init(conf, containers, isPretrain, start_fresh, *args, **kwargs)


    def policyAction(self, agentState):
        throttle = np.random.random()
        brake = np.random.random() if not agentState[2] else 0
        steer = 2*np.random.random() - 1
        toUse = "["+str(throttle)+", "+str(brake)+", "+str(steer)+"]"
        return toUse, (throttle, brake, steer) #er returned immer toUse, toSave        
        