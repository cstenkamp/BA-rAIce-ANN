import xml.etree.ElementTree as ET
import os
import numpy as np
np.set_printoptions(threshold=np.nan)
from copy import deepcopy
#====own classes====
from myprint import myprint as print

MAXSPEED = 250
DELAY_TO_CONSIDER = 100


FOLDERNAME = "SavedLaps/"
flatten = lambda l: [item for sublist in l for item in sublist]

    
    
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
                
    def make_vecs(self):
       if self.vectors != "":
           _, _, self.visionvec, self.vvec2, self.AllOneDs = cutoutandreturnvectors(self.vectors)
           self.vectors = ""
           self.flatten_oneDs()
    
    def flatten_oneDs(self):
        self.FlatOneDs = np.array(flatten(self.AllOneDs))
    
#    def normalize_oneDs(self, normalizers):
#        self.FlatOneDs -= np.array([item[0] for item in normalizers])
#        self.FlatOneDs /= np.array([item[1] for item in normalizers])
    
    def discretize_steering(self, numcats):
        self.discreteSteering = discretize_steering(self.steeringValue, numcats)

    def discretize_all(self, numcats, include_apb):#
        self.discreteAll = discretize_all(self.throttlePedalValue, self.brakePedalValue, self.discreteSteering, numcats, include_apb)

        
        
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

#input: throttle, brake, steer_AS_DISCRETE
#output: 3*speed_neurons / 4*speed_neurons                  
def discretize_all(throttle, brake, discreteSteer, numcats, include_apb):
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
#output: throttle, brake, steer
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
        return throttle, brake, steer
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
        #TODO: das dediscretize_steer und das ganze unnötige wegmachen!

   

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
            return TPList
        else:
            fraction = round(wishmsperframe/TPmsperframe*100)/100
            i = 0
            returntp = []
            while round(i) < len(TPList):
                returntp.append(TPList[round(i)])
                i += fraction
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
#        normalizers = self.find_normalizers()
        for currpoint in self.all_trackingpoints:
#            currpoint.normalize_oneDs(normalizers)
            currpoint.discretize_steering(self.steering_steps)  #TODO: numcats sollte nicht ne variable HIER sein
            #currpoint.discretize_acc_break(NUMCATS) #TODO: sollten auch andere variablen sein
            currpoint.discretize_all(self.steering_steps, self.include_accplusbreak)
            
            
#    def find_normalizers(self):
#        #was hier passiert: für jeden werte der FlatOneDs wird durchs gesamte array gegangen, das minimum gefundne, von allen subtrahiert, das maximum gefunden, dadurch geteilt.
#        normalizers = []
#        veclen = len(self.all_trackingpoints[0].FlatOneDs)
#        for i in range(veclen):
#            alle = np.array([curr.FlatOneDs[i] for curr in self.all_trackingpoints])
#            mini = min(alle)      #alle = np.subtract(alle,min(alle))
#            maxi = max(alle)-mini #alle = np.divide(alle,max(alle))
#            normalizers.append([mini,maxi])
#        return normalizers

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
    def next_batch(self, config, batch_size):
        if self.batchindex + batch_size > self.numsamples:
            raise IndexError("No more batches left")
        visions = []
        targets = []
        lookaheads = []
        speeds = []
        for indexindex in range(self.batchindex,self.batchindex+batch_size):
            i = self.randomindices[indexindex]
            vision = [self.all_trackingpoints[(i-j) % len(self.all_trackingpoints)].visionvec for j in range(config.history_frame_nr-1,-1,-1)]
            if config.use_second_camera:
                vision = vision + [self.all_trackingpoints[(i-j) % len(self.all_trackingpoints)].vvec2 for j in range(config.history_frame_nr-1,-1,-1)]
            lookahead = [self.all_trackingpoints[(i-j) % len(self.all_trackingpoints)].FlatOneDs for j in range(config.history_frame_nr-1,-1,-1)]
            target = self.all_trackingpoints[i].discreteAll
            if config.history_frame_nr == 1: 
                vision = vision[0]
                lookahead = lookahead[0]
            visions.append(np.array(vision))
            targets.append(np.array(target))
            lookaheads.append(lookahead)
            if config.speed_neurons > 0:
                speeds.append(inflate_speed(int(self.all_trackingpoints[i].speed), config.speed_neurons, config.SPEED_AS_ONEHOT))
        self.batchindex += batch_size
        return np.array(lookaheads), np.array(visions), np.array(targets), np.array(speeds)


################################################################################


def inflate_speed(speed, numberneurons, asonehot):
    speed = min(max(0,int(round(speed))), MAXSPEED)
    result = [0]*numberneurons
    if speed < 1:
        return result
    maxone = min(max(0,round((speed/MAXSPEED)*numberneurons)), numberneurons-1)
    if asonehot:
        result[maxone] = 1
    else:
        brokenspeed = round((speed/numberneurons)-maxone, 2)
        if brokenspeed < 0:
            maxone -= 1
        for i in range(maxone):
            result[i] = 1
        result[maxone] = round((speed/numberneurons)-maxone, 2)
        
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
        
    if string.find("L(") > -1:
        #print("Visionvec",self.readTwoDArrayFromString(cutout(data, "V(")))
        allOneDs.append(readOneDArrayFromString(cutout(string, "L(")))
        
    if string.find("D(") > -1:
        #print("Visionvec",self.readTwoDArrayFromString(cutout(data, "V(")))
        allOneDs.append(readOneDArrayFromString(cutout(string, "D(")))
    
    if string.find("V1(") > -1:
        #print("Visionvec",self.readTwoDArrayFromString(cutout(data, "V(")))
        visionvec = readTwoDArrayFromString(cutout(string, "V1("))  
    
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

################################################################################


    
if __name__ == '__main__':    
    import supervisedcnn
    config = supervisedcnn.Config()
    trackingpoints = TPList(FOLDERNAME, config.msperframe, config.steering_steps, config.INCLUDE_ACCPLUSBREAK)
    print("Number of samples:",trackingpoints.numsamples)
    #print(trackingpoints.all_trackingpoints[220].throttlePedalValue)
    #print(trackingpoints.all_trackingpoints[220].discreteThrottle)
    while trackingpoints.has_next(10):
        lookaheads, vision, targets, speeds = trackingpoints.next_batch(config, 10)
    print(lookaheads)
    print(targets)
    print(vision.shape)
    print(speeds)
    
    #sooo jetzt hab ich hier eine liste an trackingpoints. was ich tun muss ist jetzt dem vectors per ANN die brake, steering, throttlevalues zuzuweisen.
    
