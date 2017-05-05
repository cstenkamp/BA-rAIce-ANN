import xml.etree.ElementTree as ET
import os
import numpy as np
np.set_printoptions(threshold=np.nan)
#====own functions====
import server #from server import cutoutandreturnvectors

NUMCATS = 11



FOLDERNAME = "SavedLaps/"
flatten = lambda l: [item for sublist in l for item in sublist]


    
    
#this is supposed to resemble the TrackingPoint-Class from the recorder from Unity
class TrackingPoint(object):
    def __init__(self, time, throttlePedalValue, brakePedalValue, steeringValue, progress, vectors):
        self.time = time
        self.throttlePedalValue = float(throttlePedalValue)
        self.brakePedalValue = float(brakePedalValue)
        self.steeringValue = float(steeringValue)
        self.progress = progress
        self.vectors = vectors
                
    def make_vecs(self):
       if self.vectors != "":
           self.visionvec, self.AllOneDs = server.cutoutandreturnvectors(self.vectors)
           self.vectors = ""
           self.flatten_oneDs()
    
    def flatten_oneDs(self):
        self.FlatOneDs = np.array(flatten(self.AllOneDs))
    
    def normalize_oneDs(self, normalizers):
        self.FlatOneDs -= np.array([item[0] for item in normalizers])
        self.FlatOneDs /= np.array([item[1] for item in normalizers])
    
    
    def discretize_steering(self, numcats):
        limits = [(2/numcats)*i-1 for i in range(numcats+1)]
        limits[0] = -2
        val = numcats
        for i in range(len(limits)):
            if self.steeringValue > limits[i]:
                val = i
        self.discreteSteering = [0]*numcats
        self.discreteSteering[val] = 1
     
        
    def discretize_acc_break(self, numcats):
        def return_discrete(fromwhat):
            limits = [(1/numcats)*i for i in range(numcats+1)]
            limits[0] = -2
            val = numcats
            for i in range(len(limits)):
                if fromwhat > limits[i]:
                    val = i
            result = [0]*numcats
            result[val] = 1
            return result
          
        self.discreteThrottle = return_discrete(self.throttlePedalValue)
        self.discreteBrake = return_discrete(self.brakePedalValue)          

    def discretize_all(self):
        if self.throttlePedalValue > 0.5:
            if self.brakePedalValue > 0.5:
                self.discreteAll = [0]*(NUMCATS*3) + self.discreteSteering
            else:
                self.discreteAll = [0]*(NUMCATS*2) + self.discreteSteering + [0]*NUMCATS
        else:
            if self.brakePedalValue > 0.5:
                self.discreteAll = [0]*NUMCATS + self.discreteSteering + [0]*(NUMCATS*2)
            else:
                self.discreteAll = self.discreteSteering + [0]*(NUMCATS*3)
        
        
        
def dediscretize_steer(discrete):
    if type(discrete).__module__ == np.__name__:
        discrete = discrete.tolist()
    try:
        result = -1+(2/len(discrete))*(discrete.index(1)+0.5)
    except ValueError:
        result = 0
    return result

#probably not needed, cause its a bad idea anyway
#def dediscretize_acc_break(discrete):
#    if type(discrete).__module__ == np.__name__:
#        discrete = discrete.tolist()
#    return (1/len(discrete))*(discrete.index(1)+0.5)

def dediscretize_all(discrete):
    if type(discrete).__module__ == np.__name__:
        discrete = discrete.tolist()
    if discrete.index(1) > NUMCATS*3:
        throttle = 1
        brake = 1
        steer = dediscretize_steer(discrete[(NUMCATS*3):(NUMCATS*4)])
    elif discrete.index(1) > NUMCATS*2:
        throttle = 1
        brake = 0
        steer = dediscretize_steer(discrete[(NUMCATS*2):(NUMCATS*3)])
    elif discrete.index(1) > NUMCATS:
        throttle = 0
        brake = 1
        steer = dediscretize_steer(discrete[NUMCATS:(NUMCATS*2)])
    else:
        throttle = 0
        brake = 0
        steer = dediscretize_steer(discrete[0:NUMCATS])
    return throttle, brake, steer

   

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
            
    def __init__(self, foldername):
        assert os.path.isdir(foldername) 
        self.all_trackingpoints = []
        for file in os.listdir(foldername):
            if file.endswith(".svlap"):
                currcontent, currinfo = TPList.read_xml(os.path.join(foldername, file))
                self.all_trackingpoints.extend(currcontent)                        
                #TODO: so einfach ist das hier nicht. er hat jetzt für jedes currcontent ein currinfo, und 
                #daran sieht er alle wie viel ms getrackt wurde.... Diese function hier sollte jetzt wissen
                #alle wie viel ms der server das haben will und dementsprechend jedes x-te rauspicken..
        self.prepare_tplist()          
        self.numsamples = len(self.all_trackingpoints)
        self.reset_batch()

    def prepare_tplist(self):
        for currpoint in self.all_trackingpoints:
            currpoint.make_vecs();     
        normalizers = self.find_normalizers()
        for currpoint in self.all_trackingpoints:
            currpoint.normalize_oneDs(normalizers)
            currpoint.discretize_steering(NUMCATS)  #TODO: numcats sollte nicht ne variable HIER sein
            #currpoint.discretize_acc_break(NUMCATS) #TODO: sollten auch andere variablen sein
            currpoint.discretize_all()
            
            
    def find_normalizers(self):
        #was hier passiert: für jeden werte der FlatOneDs wird durchs gesamte array gegangen, das minimum gefundne, von allen subtrahiert, das maximum gefunden, dadurch geteilt.
        normalizers = []
        veclen = len(self.all_trackingpoints[0].FlatOneDs)
        for i in range(veclen):
            alle = np.array([curr.FlatOneDs[i] for curr in self.all_trackingpoints])
            mini = min(alle)      #alle = np.subtract(alle,min(alle))
            maxi = max(alle)-mini #alle = np.divide(alle,max(alle))
            normalizers.append([mini,maxi])
        return normalizers

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
        discretetargets = []
        for indexindex in range(self.batchindex,self.batchindex+batch_size):
            i = self.randomindices[indexindex]
            vision = [self.all_trackingpoints[(i-j) % len(self.all_trackingpoints)].visionvec for j in range(config.history_frame_nr-1,-1,-1)]
            lookahead = [self.all_trackingpoints[(i-j) % len(self.all_trackingpoints)].FlatOneDs for j in range(config.history_frame_nr-1,-1,-1)]
            #target = [self.all_trackingpoints[i].throttlePedalValue, self.all_trackingpoints[i].brakePedalValue, self.all_trackingpoints[i].steeringValue]
            target = self.all_trackingpoints[i].discreteAll
            #discretetarget = flatten([self.all_trackingpoints[i].discreteThrottle, self.all_trackingpoints[i].discreteBrake, self.all_trackingpoints[i].discreteSteering])
            discretetarget = flatten([[0]*(NUMCATS*2), self.all_trackingpoints[i].discreteSteering])
            if config.history_frame_nr == 1: 
                vision = vision[0]
                lookahead = lookahead[0]
            visions.append(np.array(vision))
            targets.append(np.array(target))
            discretetargets.append(np.array(discretetarget))
            lookaheads.append(lookahead)
        self.batchindex += batch_size
        return np.array(lookaheads), np.array(visions), np.array(targets), np.array(discretetargets)




if __name__ == '__main__':    
    import supervisedcnn
    config = supervisedcnn.Config()
    trackingpoints = TPList(FOLDERNAME)
    print("Number of samples:",trackingpoints.numsamples)
    #print(trackingpoints.all_trackingpoints[220].throttlePedalValue)
    #print(trackingpoints.all_trackingpoints[220].discreteThrottle)
    while trackingpoints.has_next(10):
        lookaheads, vision, targets, dtargets = trackingpoints.next_batch(config, 10)
    print(lookaheads)
    print(dtargets[:,22:])
    print(targets)
    print(vision.shape)
    
    
    #sooo jetzt hab ich hier eine liste an trackingpoints. was ich tun muss ist jetzt dem vectors per ANN die brake, steering, throttlevalues zuzuweisen.
    