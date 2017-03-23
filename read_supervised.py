import xml.etree.ElementTree as ET

FOLDERNAME = "SavedLaps/"

#TODO ne funktion die alle XMLs aus dem Ordner SavedLaps ausliest
    




def read_xml(FileName):
    tree = ET.parse(FileName)
    root = tree.getroot()
    assert root.tag=="ArrayOfTrackingPoint", "that is not the kind of XML I thought it would be."
    
    

if __name__ == '__main__':
    read_xml(FOLDERNAME+"complete_17_03_22__03_52_12.svlap")