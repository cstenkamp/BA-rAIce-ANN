import numpy as np
#====own classes====
from agent import AbstractAgent

class Agent(AbstractAgent):    
    def __init__(self, conf, containers, isPretrain=False, start_fresh=False, *args, **kwargs):
        self.name = "random_agent"
        super().__init__(conf, containers, isPretrain, start_fresh, *args, **kwargs)
        self.isSupervised = True
        

    def policyAction(self, agentState):
        throttle = np.random.random()
        brake = np.random.random() if not agentState[2] else 0
        steer = 2*np.random.random() - 1
        toUse = "["+str(throttle)+", "+str(brake)+", "+str(steer)+"]"
        return toUse, (throttle, brake, steer) #er returned immer toUse, toSave        
        
    def handle_commands(self, command, wasValid=False):
        self.eval_episodeVals(command)
        if command == "turnedaround":
            self.resetUnityAndServer()
        if command == "wallhit":   
            self.resetUnityAndServer()
        if command == "lapdone":
            self.resetUnityAndServer()
