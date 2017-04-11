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
     
    @staticmethod
    def dediscretize_steering(discrete_steer):
        return -1+(2/len(discrete_steer))*(discrete_steer.index(1)+0.5)

   

class TPList(object):
    
    @staticmethod
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
            
    def __init__(self, foldername):
        assert os.path.isdir(foldername) 
        self.all_trackingpoints = []
        for file in os.listdir(foldername):
            if file.endswith(".svlap"):
                self.all_trackingpoints.extend(TPList.read_xml(os.path.join(foldername, file)))
        self.prepare_tplist()          
        self.numsamples = len(self.all_trackingpoints)
        self.reset_batch()

    def prepare_tplist(self):
        for currpoint in self.all_trackingpoints:
            currpoint.make_vecs();     
        normalizers = self.find_normalizers()
        for currpoint in self.all_trackingpoints:
            currpoint.normalize_oneDs(normalizers)
            currpoint.discretize_steering(NUMCATS)
            
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
        for indexindex in range(self.batchindex,self.batchindex+batch_size):
            i = self.randomindices[indexindex]
            vision = [self.all_trackingpoints[(i-j) % len(self.all_trackingpoints)].visionvec for j in range(config.history_frame_nr,-1,-1)]
            lookahead = [self.all_trackingpoints[(i-j) % len(self.all_trackingpoints)].FlatOneDs for j in range(config.history_frame_nr,-1,-1)]
            target = [self.all_trackingpoints[i].throttlePedalValue, self.all_trackingpoints[i].brakePedalValue, self.all_trackingpoints[i].steeringValue, self.all_trackingpoints[i].discreteSteering]
            if config.history_frame_nr == 1: 
                vision = vision[0]
                lookahead = lookahead[0]
            visions.append(np.array(vision))
            targets.append(np.array(target[3]))
            lookaheads.append(lookahead)
        self.batchindex += batch_size
        return np.array(lookaheads), np.array(visions), np.array(targets)

             

            





#def sample_batch(batch_size, config, dataset):
#    indices = np.random.choice(len(dataset), batch_size)
#    visions = []
#    targets = []
#    lookaheads = []
#    for i in indices:
#        vision = [dataset[(i-j) % len(dataset)].visionvec for j in range(config.history_frame_nr,-1,-1)]
#        lookahead = [dataset[(i-j) % len(dataset)].FlatOneDs for j in range(config.history_frame_nr,-1,-1)]
#        target = [dataset[i].throttlePedalValue, dataset[i].brakePedalValue, dataset[i].steeringValue, dataset[i].discreteSteering]
#        if config.history_frame_nr == 1: 
#            vision = vision[0]
#            lookahead = lookahead[0]
#        visions.append(vision)
#        targets.append(target[3])
#        lookaheads.append(lookahead)
#    return lookaheads, visions, targets





#nächste schritte
# -dafür sorgen dass das irgendwas sinvolles tut
# -weights speichern
# -den output anzeigen lassen können


if __name__ == '__main__':    
    import supervisedcnn
    config = supervisedcnn.Config()
    trackingpoints = TPList(FOLDERNAME)
    print("Number of samples:",trackingpoints.numsamples)
#    while trackingpoints.has_next(10):
#        lookaheads, _, targets = trackingpoints.next_batch(config, 10)
#    print(lookaheads)
#    print(targets)
#    
    
    
    #sooo jetzt hab ich hier eine liste an trackingpoints. was ich tun muss ist jetzt dem vectors per ANN die brake, steering, throttlevalues zuzuweisen.
    