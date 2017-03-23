import xml.etree.ElementTree as ET

FOLDERNAME = "SavedLaps/"

#TODO ne funktion die alle XMLs aus dem Ordner SavedLaps ausliest
    
#this is supposed to resemble the TrackingPoint-Class from the recorder from Unity
class TrackingPoint(object):
    def __init__(self, time, throttlePedalValue, brakePedalValue, steeringValue, progress, vectors):
        self.time = time
        self.throttlePedalValue = throttlePedalValue
        self.brakePedalValue = brakePedalValue
        self.steeringValue = steeringValue
        self.progress = progress
        self.vectors = vectors
    
        
        

def read_xml(FileName):
    all_trackingpoints = []
    tree = ET.parse(FileName)
    root = tree.getroot()
    assert root.tag=="ArrayOfTrackingPoint", "that is not the kind of XML I thought it would be."
    for currpoint in root:
        inputdict = {}
        for item in currpoint:
            inputdict[item.tag] = item.text
        tp = TrackingPoint(**inputdict) #ein dictionary mit kwargs, IM SO PYTHON!!
        all_trackingpoints.append(tp)
    return all_trackingpoints


#sooo jetzt hab ich hier eine liste an trackingpoints. was ich tun muss ist jetzt dem vectors per ANN die brake, steering, throttlevalues zuzuweisen.

        
        
if __name__ == '__main__':
    read_xml(FOLDERNAME+"complete_17_03_22__03_52_12.svlap")