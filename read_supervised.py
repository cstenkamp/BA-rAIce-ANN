import xml.etree.ElementTree as ET
import os
import numpy as np
np.set_printoptions(threshold=np.nan)
from copy import deepcopy
from math import floor
from collections import namedtuple
#====own classes====
from myprint import myprint as print
from config import Config

flatten = lambda l: [item for sublist in l for item in sublist]
DELAY_TO_CONSIDER = 100

###############################################################################
###############################################################################

#this very long part is the comparable namedtuple otherinputs!
Preprogressvec = namedtuple('ProgressVec', ['Progress', 'Laptime', 'NumRounds', 'fValidLap'])
Prespeedsteer = namedtuple('SpeedSteer', ['RLTorque', 'RRTorque', 'FLSteer', 'FRSteer', 'velocity', 'rightDirection', 'velocityOfPerpendiculars', 'carAngle', 'speedInStreetDir','speedInTraverDir', 'CurvinessBeforeCar'])
Prestatusvector = namedtuple('StatusVector', ['FLSlip0', 'FRSlip0', 'RLSlip0', 'RRSlip0', 'FLSlip1', 'FRSlip1', 'RLSlip1', 'RRSlip1'])
                                             #4 elems       11 elems      8 elems         1 elem        15 elems         7 elems        30 elems       2 elems    3 elems  = 81 elems
Preotherinputs = namedtuple('OtherInputs', ['ProgressVec', 'SpeedSteer', 'StatusVector', 'CenterDist', 'CenterDistVec', 'WallDistVec', 'LookAheadVec', 'FBDelta', 'Action'])
class Progressvec(Preprogressvec):
    def __eq__(self, other):
        return np.all([self[i] == other[i] for i in [0,1,2]]) #Zeit wird nicht berücksichtigt (wenn doch ",3" hinzufügen)
class Speedsteer(Prespeedsteer):
    def __eq__(self, other):
        return np.all([self[i] == other[i] for i in range(len(self))])
class Statusvector(Prestatusvector):
    def __eq__(self, other):
        return np.all([self[i] == other[i] for i in range(len(self))])
class Otherinputs(Preotherinputs):
    def __eq__(self, other):
        if other == None:
            return self.empty()
        return self.ProgressVec == other.ProgressVec \
           and self.SpeedSteer ==  other.SpeedSteer \
           and self.StatusVector == other.StatusVector \
           and self.CenterDist == other.CenterDist \
           and np.all(self.WallDistVec == other.WallDistVec) \
           and np.all(self.LookAheadVec == other.LookAheadVec)
           #and np.all(self.Action == other.Action) #? macht das sinn? #TODO: is the performed action relevant??? hmm.
           #and np.all(self.CenterDistVec == other.CenterDistVec) \ #can be skipped because then the centerdist is also equal
           #FBDelta werden auch nicht beachtet, da die ebenfalls von Zeit abhängen 
    def empty(self):
        return self.__eq__(empty_inputs())
    def returnRelevant(self):
        print("Removed 4 elements from speedsteer here, seems necessary", level=-1)
        return [i for i in self.CenterDistVec]+[0]*4+[i for i in self.SpeedSteer[4:]]+[i for i in self.StatusVector]+[i for i in self.WallDistVec]+[i for i in self.LookAheadVec]
    def as_list(self):
        return [list(self.ProgressVec), list(self.SpeedSteer), list(self.StatusVector), self.CenterDist+self.CenterDistVec, self.WallDistVec, self.LookAheadVec, self.FBDelta, self.Action]
    def normalized(self):
        x = self.as_list()
#        tmp = flatten([[((x[i][j] - MINVALS.as_list()[i][j])/ (MAXVALS.as_list()[i][j]-MINVALS.as_list()[i][j])) for j in range(len(x[i]))] for i in range(len(x))])
#        for i in tmp[2:]:
#            if abs(i) > 1.2:
#                print(make_otherinputs([[((x[i][j] - MINVALS.as_list()[i][j])/ (MAXVALS.as_list()[i][j]-MINVALS.as_list()[i][j])) for j in range(len(x[i]))] for i in range(len(x))]),level=100)
        return make_otherinputs([[((x[i][j] - MINVALS.as_list()[i][j])/ (MAXVALS.as_list()[i][j]-MINVALS.as_list()[i][j])) for j in range(len(x[i]))] for i in range(len(x))])
                      
empty_progressvec = lambda: Progressvec(0, 0, 0, 0)
empty_speedsteer = lambda: Speedsteer(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
empty_statusvector = lambda: Statusvector(0, 0, 0, 0, 0, 0, 0, 0)
empty_inputs = lambda: Otherinputs(empty_progressvec(), empty_speedsteer(), empty_statusvector(), [0], np.zeros(15), np.zeros(7), np.zeros(30), np.zeros(2), np.zeros(3))
def make_otherinputs(othervecs):
    return Otherinputs(Progressvec(othervecs[0][0], othervecs[0][1], othervecs[0][2], othervecs[0][3]), \
                       Speedsteer(othervecs[1][0], othervecs[1][1], othervecs[1][2], othervecs[1][3], othervecs[1][4], othervecs[1][5], othervecs[1][6], othervecs[1][7], othervecs[1][8], othervecs[1][9], othervecs[1][10]), \
                       Statusvector(othervecs[2][0], othervecs[2][1], othervecs[2][2], othervecs[2][3], othervecs[2][4], othervecs[2][5], othervecs[2][6], othervecs[2][7]), \
                       [othervecs[3][0]], \
                       othervecs[3][1:], \
                       othervecs[4], 
                       othervecs[5], 
                       othervecs[6],
                       othervecs[7])
    
maxspeed = Config().MAXSPEED                                                   
MINVALS = Otherinputs(Progressvec(0,0,0,0), Speedsteer(0,0,-20,-20,0,0,0,-180,0,-maxspeed,-1),Statusvector(-5,-5,-5,-5,-5,-5,-5,-5),[-13],[0]*15,[0]*7,[-52]*30,[-60]*2,[0 for i in Config().action_bounds])
MAXVALS = Otherinputs(Progressvec(100,1,100,1), Speedsteer(1200,1200,20,20,maxspeed,1,maxspeed,180,maxspeed,maxspeed,1),Statusvector(5,5,5,5,5,5,5,5),[13],[0.3989]*15,[300]*7,[52]*30,[60]*2,[1 for i in Config().action_bounds])
#this very long part end

###############################################################################
###############################################################################
    
#this is supposed to resemble the TrackingPoint-Class from the recorder from Unity
class TrackingPoint(object):
    def __init__(self, time, throttlePedalValue, brakePedalValue, steeringValue, progress, vectors, speed):
        self.time = time
        self.throttlePedalValue = float(throttlePedalValue)
        self.brakePedalValue = float(brakePedalValue)
        self.steeringValue = float(steeringValue)
        self.progress = progress
        self.vectors = vectors
        self.speed = speed
        self.endedAfter = False
                
    def make_vecs(self):
       if self.vectors != "":
           _, _, self.visionvec, self.vvec2, othervecs = cutoutandreturnvectors(self.vectors)
           self.vectors = ""
           self.otherinputs = make_otherinputs(othervecs).normalized()
           
    def discretize_all(self, numcats, include_apb):
        self.discreteAll = discretize_all(self.throttlePedalValue, self.brakePedalValue, self.steeringValue, numcats, include_apb)

        
###############################################################################
###############################################################################

#input: throttle, brake, steer as int, int, real
#output: 3*speed_neurons / 4*speed_neurons                  
def discretize_all(throttle, brake, steer, numcats, include_apb):
    def discretize_steering(steeringVal, numcats):
        limits = [(2/numcats)*i-1 for i in range(numcats+1)]
        limits[0] = -2
        val = numcats
        for i in range(len(limits)):
            if steeringVal > limits[i]:
                val = i
        discreteSteering = [0]*numcats
        discreteSteering[val] = 1     
        return discreteSteering      
    
    discreteSteer = discretize_steering(steer, numcats)
    
    if include_apb:
        if throttle > 0.5:
            if brake > 0.5:
                discreteAll = [0]*(numcats*3) + discreteSteer
            else:
                discreteAll = [0]*(numcats*2) + discreteSteer + [0]*numcats
        else:
            if brake > 0.5:
                discreteAll = [0]*numcats + discreteSteer + [0]*(numcats*2)
            else:
                discreteAll = discreteSteer + [0]*(numcats*3)
    else:
        if brake > 0.5:
            discreteAll = [0]*(numcats*2) + discreteSteer
        else:
            if throttle > 0.5:
                discreteAll = [0]*numcats + discreteSteer + [0]*(numcats)
            else:
                discreteAll = discreteSteer + [0]*(numcats*2)                
    return discreteAll
               
 
def dediscretize_steer(discrete):
    if type(discrete).__module__ == np.__name__:
        discrete = discrete.tolist()
    try:
        result = round(-1+(2/len(discrete))*(discrete.index(1)+0.5), 3)
    except ValueError:
        result = 0
    return result    
        

#input:  3*speed_neurons / 4*speed_neurons
#output: throttle, brake, steer (as int, int, real)
def dediscretize_all(discrete, numcats, include_apb):
    if type(discrete).__module__ == np.__name__:
        discrete = discrete.tolist()
        
    if include_apb:
        if discrete.index(1) >= numcats*3:
            throttle = 1
            brake = 1
            steer = dediscretize_steer(discrete[(numcats*3):(numcats*4)])
        elif discrete.index(1) >= numcats*2:
            throttle = 1
            brake = 0
            steer = dediscretize_steer(discrete[(numcats*2):(numcats*3)])
        elif discrete.index(1) >= numcats:
            throttle = 0
            brake = 1
            steer = dediscretize_steer(discrete[numcats:(numcats*2)])
        else:
            throttle = 0
            brake = 0
            steer = dediscretize_steer(discrete[0:numcats])
    else:
        if discrete.index(1) >= numcats*2:
            throttle = 0
            brake = 1
            steer = dediscretize_steer(discrete[(numcats*2):(numcats*3)])
        elif discrete.index(1) >= numcats:
            throttle = 1
            brake = 0
            steer = dediscretize_steer(discrete[numcats:(numcats*2)])
        else:
            throttle = 0
            brake = 0
            steer = dediscretize_steer(discrete[0:numcats])
    return throttle, brake, steer

   
###############################################################################
###############################################################################


class TPList(object):
    
    @staticmethod
    def read_xml(FileName):
        this_trackingpoints = []
        furtherinfo = {}
        tree = ET.parse(FileName)
        root = tree.getroot()
        assert root.tag=="TPMitInfoList", "that is not the kind of XML I thought it would be."
        for majorpoint in root:
            if majorpoint.tag == "TPList":
                for currpoint in majorpoint:
                    inputdict = {}
                    for item in currpoint:
                        inputdict[item.tag] = item.text
                    tp = TrackingPoint(**inputdict) #ein dictionary mit kwargs, IM SO PYTHON!!
                    this_trackingpoints.append(tp)
            else:
                furtherinfo[majorpoint.tag] = majorpoint.text
        return this_trackingpoints, furtherinfo
    
    def __init__(self, foldername, twocams, msperframe, steering_steps, include_accplusbreak):
        assert os.path.isdir(foldername) 
        self.all_trackingpoints = []
        self.steering_steps = steering_steps
        self.include_accplusbreak = include_accplusbreak
        for file in os.listdir(foldername):
            if file.endswith(".svlap") and (("2cam" in file) if twocams else ("1cam" in file)):
                currcontent, currinfo = TPList.read_xml(os.path.join(foldername, file))
                if DELAY_TO_CONSIDER > 0:
                    currcontent = self.consider_delay(currcontent, int(currinfo["trackAllXMS"]))
                currcontent = self.extract_appropriate(currcontent, int(currinfo["trackAllXMS"]), msperframe, currinfo["filename"])    
                if currcontent is not None:                 
                    self.all_trackingpoints.extend(currcontent)
                    
        self.prepare_tplist()          
        self.numsamples = len(self.all_trackingpoints)
        self.reset_batch()

    def extract_appropriate(self, TPList, TPmsperframe, wishmsperframe, filename):
        if float(TPmsperframe) > float(wishmsperframe)*1.05:
            print("%s could not be used because it recorded not enough frames!" % filename)
            return None
        elif float(wishmsperframe)*0.95 < float(TPmsperframe) < float(wishmsperframe)*1.05:
            returntp = TPList
        else:
            fraction = round(wishmsperframe/TPmsperframe*100)/100
            i = 0
            returntp = []
            while round(i) < len(TPList):
                returntp.append(TPList[round(i)])
                i += fraction
        returntp[len(returntp)-1].endedAfter = True
        return returntp
    
    def consider_delay(self, TPList, TPmsperframe):
        result = deepcopy(TPList)
        for i in range(len(TPList)):
            j = max(i-(DELAY_TO_CONSIDER//TPmsperframe),0)
            #the server is a bit delayed. We consider that by mapping the current vision to the output a few frames ago.
            result[i].brakePedalValue = TPList[j].brakePedalValue #älterer output (output von j), neuerer vision (von i)!
            result[i].throttlePedalValue = TPList[j].throttlePedalValue
            result[i].steeringValue = TPList[j].steeringValue
        return result
    
        
    def prepare_tplist(self):
        for currpoint in self.all_trackingpoints:
            currpoint.make_vecs();     
            currpoint.discretize_all(self.steering_steps, self.include_accplusbreak)
            

    def reset_batch(self):
        self.batchindex = 0
        self.randomindices = np.random.permutation(self.numsamples)
        
    def has_next(self, batch_size):
        return self.batchindex + batch_size <= self.numsamples
        
    def num_batches(self, batch_size):
        return self.numsamples//batch_size
        
    
    #TODO: sample according to information gain, what DQN didn't do yet.
    #TODO: uhm, das st jezt simples ziehen mit zurücklegen, every time... ne richtige next_batch funktion, bei der jedes mal vorkommt wäre sinnvoller, oder?
    #TODO: splitting into training and validation set??    
    def next_batch(self, conf, agent, batch_size):
        if self.batchindex + batch_size > self.numsamples:
            raise IndexError("No more batches left")
            
        vvec_batch = []
        past_vvec_batch = []
        vvec2_batch = []
        past_vvec2_batch = []
        otherinputs_batch = []
        past_otherinputs_batch = []
        actions_batch = []
        past_actions_batch = []
        episode_ended_batch = []
        
        for indexindex in range(self.batchindex,self.batchindex+batch_size):
            
            vvec1_hist, vvec2_hist, otherinput_hist, action_hist = self._read(conf, agent, batch_size, self.randomindices[indexindex])
            past_vvec1_hist, past_vvec2_hist, past_otherinput_hist, past_action_hist = self._read(conf, agent, batch_size, self.randomindices[indexindex], readPast = True)
            episode_ended_batch.append(self.all_trackingpoints[((self.randomindices[indexindex]-1) % len(self.all_trackingpoints))].endedAfter)
            
            vvec_batch.append(vvec1_hist)
            past_vvec_batch.append(past_vvec1_hist)
            vvec2_batch.append(vvec2_hist)
            past_vvec2_batch.append(past_vvec2_hist)
            otherinputs_batch.append(otherinput_hist)
            past_otherinputs_batch.append(past_otherinput_hist)
            actions_batch.append(action_hist)
            past_actions_batch.append(past_action_hist)

        self.batchindex += batch_size
        return (np.array(vvec_batch), np.array(vvec2_batch), otherinputs_batch, np.array(actions_batch)), \
               (np.array(past_vvec_batch), np.array(past_vvec2_batch), past_otherinputs_batch, np.array(past_actions_batch)), \
               np.array(episode_ended_batch)

    
    def _read(self, conf, agent, batch_size, index, readPast = False):
            vh = [] if agent.usesConv else None
            v2h = [] if agent.usesConv and conf.use_second_camera else None
            oih = []
            ah = []
            for j in range(conf.history_frame_nr-1,-1,-1):
                index = (index-j-1) % len(self.all_trackingpoints) if readPast else (index-j) % len(self.all_trackingpoints)
                if agent.usesConv:
                    vh.append(self.all_trackingpoints[index].visionvec)
                    if conf.use_second_camera:
                        v2h.append(self.all_trackingpoints[index].vvec2)
                oih.append(self.all_trackingpoints[index].otherinputs)
                ah.append((self.all_trackingpoints[index].throttlePedalValue, self.all_trackingpoints[index].brakePedalValue, self.all_trackingpoints[index].steeringValue))
            return np.array(vh), np.array(v2h), oih, np.array(ah)


    
    #returns [[s],[a],[r],[s2],[t]], where every state s = (conf, ff)
    @staticmethod
    def create_QLearnInputs_fromBatch(presentStates, pastStates, resetAfters, agent):
        presentStates = list(zip(*presentStates))
        pastStates = list(zip(*pastStates))
        old_convs =  np.rollaxis(np.array([agent.getAgentState(*i)[0] for i in pastStates]), 1, 4) if agent.usesConv else [None]*len(pastStates)
        old_other = np.array([agent.makeNetUsableOtherInputs(agent.getAgentState(*i)[1]) for i in pastStates])
        oldAgentStates = list(zip(old_convs, old_other))
        actions = [agent.makeNetUsableAction(agent.getAction(*i)) for i in pastStates]   
        if not agent.isSupervised:               
            new_convs = np.rollaxis(np.array([agent.getAgentState(*i)[0] for i in presentStates]), 1, 4) if agent.usesConv else [None]*len(presentStates)
            new_other = np.array([agent.makeNetUsableOtherInputs(agent.getAgentState(*i)[1]) for i in presentStates])
            newAgentStates = list(zip(new_convs, new_other))
            rewards = [agent.calculateReward(*i) for i in pastStates]
        else:
            newAgentStates, rewards, resetAfters = None, None, None
        
        return oldAgentStates, np.array(actions), np.array(rewards), newAgentStates, np.array(resetAfters) #wurde angepasst auf s,a,r,s2,t



###############################################################################
###############################################################################

def inflate_speed(speed, numberneurons, asonehot, maxspeed):
    speed = min(max(0,int(round(speed))), maxspeed)
    result = [0]*numberneurons
    if speed < 1:
        return result
    maxone = min(max(0,floor((speed/maxspeed)*numberneurons)), numberneurons-1)
    if asonehot:
        result[maxone] = 1
    else:
        brokenspeed = round((speed - (maxone/numberneurons*maxspeed)) / (maxspeed/numberneurons), 2)
    
        for i in range(maxone):
            result[i] = 1
        result[maxone] = brokenspeed
        
    return result

        
def cutoutandreturnvectors(string):
    allOneDs  = []
    visionvec = [[]]    
    STime = 0
    CTime = 0
    def cutout(string, letter):
        return string[string.find(letter)+len(letter):string[string.find(letter):].find(")")+string.find(letter)]
    
    if string.find("STime") > -1:
        STime = int(cutout(string, "STime("))        
    
    if string.find("CTime") > -1:
        CTime = int(cutout(string, "CTime("))        
    
    if string.find("P(") > -1:
        allOneDs.append(readOneDArrayFromString(cutout(string, "P(")))

    if string.find("S(") > -1:
        #print("SpeedStearVec",self.readOneDArrayFromString(cutout(data, "S(")))
        allOneDs.append(readOneDArrayFromString(cutout(string, "S(")))

    if string.find("T(") > -1:
        #print("CarStatusVec",self.readOneDArrayFromString(cutout(data, "T(")))
        allOneDs.append(readOneDArrayFromString(cutout(string, "T(")))
        
    if string.find("C(") > -1:
        #print("Visionvec",self.readTwoDArrayFromString(cutout(data, "V(")))
        allOneDs.append(readOneDArrayFromString(cutout(string, "C(")))
        
    if string.find("W(") > -1:
        #print("Visionvec",self.readTwoDArrayFromString(cutout(data, "V(")))
        allOneDs.append(readOneDArrayFromString(cutout(string, "W(")))
        
    if string.find("L(") > -1:
        #print("Visionvec",self.readTwoDArrayFromString(cutout(data, "V(")))
        allOneDs.append(readOneDArrayFromString(cutout(string, "L(")))
        
    if string.find("D(") > -1:
        #print("Visionvec",self.readTwoDArrayFromString(cutout(data, "V(")))
        allOneDs.append(readOneDArrayFromString(cutout(string, "D(")))
    
    if string.find("A(") > -1:
        #print("Visionvec",self.readTwoDArrayFromString(cutout(data, "V(")))
        allOneDs.append(readOneDArrayFromString(cutout(string, "A(")))
    
    if string.find("V1(") > -1:
        #print("Visionvec",self.readTwoDArrayFromString(cutout(data, "V(")))
        visionvec = readTwoDArrayFromString(cutout(string, "V1("))  
    else:
        visionvec = None
    
    if string.find("V2(") > -1:
        #print("Visionvec",self.readTwoDArrayFromString(cutout(data, "V(")))
        vvec2 = readTwoDArrayFromString(cutout(string, "V2("))    
    else:
        vvec2 = None
        
    return STime, CTime, visionvec, vvec2, allOneDs
        

def readOneDArrayFromString(string):
    tmpstrings = string.split(",")
    tmpfloats = []
    for i in tmpstrings:
        tmp = i.replace(" ","")
        if len(tmp) > 0:
            try:
                tmp = ("1" if tmp == "T" else "0" if tmp == "F" else tmp)
                x = float(str(tmp))
                tmpfloats.append(x)  
            except ValueError:
                print("I'm crying") #cry. 
    return tmpfloats


def ternary(n):
    if n == 0:
        return '0'
    nums = []
    if n < 0:
        n*=-1
    while n:
        n, r = divmod(n, 3)
        nums.append(str(r))
    return ''.join(reversed(nums))


def readTwoDArrayFromString(string):
    tmpstrings = string.split(",")
    tmpreturn = []
    for i in tmpstrings:
        tmp = i.replace(" ","")
        if len(tmp) > 0:
            try:
                currline = []
                for j in tmp:
                    currline.append(int(j))
                tmpreturn.append(currline)
            except ValueError:
                print("I'm crying") #cry.
    return np.array(tmpreturn)

    


###############################################################################
###############################################################################


if __name__ == '__main__':    
    import config
    conf = config.Config()
    import dqn_rl_agent
    from server import Containers
    myAgent = dqn_rl_agent.Agent(conf, Containers(), True)
    trackingpoints = TPList(conf.LapFolderName, conf.use_second_camera, conf.msperframe, conf.steering_steps, conf.INCLUDE_ACCPLUSBREAK)
    print("Number of samples:",trackingpoints.numsamples)
    while trackingpoints.has_next(10):
        presentStates, pastStates, _ = trackingpoints.next_batch(conf, myAgent, 10)
    
    vvec, vvec2, otherinputs, actions = presentStates  
    print(vvec.shape)
    print(vvec2.shape)
    print(len(otherinputs))
    print(actions.shape)  
    
    s, a, r, s2, t = create_QLearnInputs_from_PTStateBatch(presentStates, pastStates, myAgent)
    
    print(a)
        