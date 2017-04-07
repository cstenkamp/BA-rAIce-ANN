import xml.etree.ElementTree as ET
import os
import numpy as np
np.set_printoptions(threshold=np.nan)
#====own functions====
from server import cutoutandreturnvectors

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
           self.visionvec, self.AllOneDs = cutoutandreturnvectors(self.vectors)
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
        
        
                  
    
    
def read_all_xmls(foldername):
    assert os.path.isdir(foldername) 
    all_trackingpoints = []
    for file in os.listdir(foldername):
        if file.endswith(".svlap"):
            all_trackingpoints.extend(read_xml(os.path.join(foldername, file)))
    all_trackingpoints = prepare_tplist(all_trackingpoints)
    return all_trackingpoints        


            
def read_xml(FileName):
    this_trackingpoints = []
    tree = ET.parse(FileName)
    root = tree.getroot()
    assert root.tag=="ArrayOfTrackingPoint", "that is not the kind of XML I thought it would be."
    for currpoint in root:
        inputdict = {}
        for item in currpoint:
            inputdict[item.tag] = item.text
        tp = TrackingPoint(**inputdict) #ein dictionary mit kwargs, IM SO PYTHON!!
        this_trackingpoints.append(tp)
    return this_trackingpoints


def prepare_tplist(all_trackingpoints):
    for currpoint in all_trackingpoints:
        currpoint.make_vecs();     
    normalizers = find_normalizers(all_trackingpoints)
    for currpoint in all_trackingpoints:
        currpoint.normalize_oneDs(normalizers)
        currpoint.discretize_steering(NUMCATS)
    return all_trackingpoints


def find_normalizers(trackingpoints):
    #was hier passiert: für jeden werte der FlatOneDs wird durchs gesamte array gegangen, das minimum gefundne, von allen subtrahiert, das maximum gefunden, dadurch geteilt.
    normalizers = []
    veclen = len(trackingpoints[0].FlatOneDs)
    for i in range(veclen):
        alle = np.array([curr.FlatOneDs[i] for curr in trackingpoints])
        mini = min(alle)      #alle = np.subtract(alle,min(alle))
        maxi = max(alle)-mini #alle = np.divide(alle,max(alle))
        normalizers.append([mini,maxi])
    return normalizers



#TODO: sample according to information gain, what DQN didn't do yet.
#TODO: uhm, das st jezt simples ziehen mit zurücklegen, every time... ne richtige next_batch funktion, bei der jedes mal vorkommt wäre sinnvoller, oder?
def sample_batch(config, dataset, visions=True):
    indices = np.random.choice(len(dataset), config.batch_size)
    visions = []
    targets = []
    lookaheads = []
    for i in indices:
        vision = [dataset[(i-j) % len(dataset)].visionvec for j in range(config.history_frame_nr,-1,-1)]
        lookahead = [dataset[(i-j) % len(dataset)].FlatOneDs for j in range(config.history_frame_nr,-1,-1)]
        target = [dataset[i].throttlePedalValue, dataset[i].brakePedalValue, dataset[i].steeringValue, dataset[i].discreteSteering]
        if config.history_frame_nr == 1: 
            vision = vision[0]
            lookahead = lookahead[0]
        visions.append(vision)
        targets.append(target[3])
        lookaheads.append(lookahead)
    return lookaheads, visions, targets







#nächste schritte
# -dafür sorgen dass das irgendwas sinvolles tut
# -weights speichern
# -den output anzeigen lassen können




if __name__ == '__main__':    
    import supervisedcnn
    config = supervisedcnn.Config()
    all_trackingpoints = read_all_xmls(FOLDERNAME)
    print("Number of samples:",len(all_trackingpoints))
    lookaheads, _, targets = sample_batch(config, all_trackingpoints, False)
    print(lookaheads)
    print(targets)
    
    
    
    #sooo jetzt hab ich hier eine liste an trackingpoints. was ich tun muss ist jetzt dem vectors per ANN die brake, steering, throttlevalues zuzuweisen.
    