# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 15:36:29 2017

@author: csten_000
"""
import matplotlib
matplotlib.use('WXAgg') #such that plots run in side threads (so pip install wxpython)
import matplotlib.pyplot as plt
import numpy as np
import threading 
from math import sqrt, ceil
import xml.etree.ElementTree as ET
from xml.dom import minidom
import datetime
import os
#====own classes====
from utils import run_from_ipython
flatten = lambda l: [item for sublist in l for item in sublist]
def flatten2(l):
    try:
        return flatten(l)
    except:
        return l

SHOW_IN_SUBPLOTS = True

class evaluator():
    def __init__(self, containers, agent, show_plot, save_xml, labels, val_bounds): 
        self.save_xml = save_xml
        self.show_plot = show_plot
        self.internal_num = 0
        
        if self.save_xml:
            self.xml_saver = xml_saver(containers, agent, labels)
            
        if self.show_plot:
            self.plotter = plotter(agent.name, labels, val_bounds)
        

            
    def _add_episode(self, new_vals, **kwargs):
        if self.save_xml:
            self.xml_saver.add_episode(new_vals, **kwargs)
            self.xml_saver.save()
        
        if self.show_plot:
            self.plotter.update(*new_vals) 


    def add_episode(self, *args, **kwargs):
        if not "nr" in kwargs:
            self.internal_num += 1
            kwargs["nr"] = self.internal_num
        t1 = threading.Thread(target=self._add_episode, args=(args), kwargs=(kwargs))
        t1.start()


#    def add_targetnetcopy(self, **kwargs):
#        if self.save_xml:
#            self.xml_saver.add_targetnetcopy(**kwargs)
#            self.xml_saver.save()      






###############################################################################
class xml_saver():
    def __init__(self, containers, agent, labels):
        self.containers = containers
        self.agent = agent
        self.agentname = agent.name
        self.xmlfilename = agent.folder(agent.conf.xml_dir)+self.agentname+"_eval.xml"
        self.labels = [i.replace(" ","_",) for i in labels]
        self.episode = 0    
        self.root, self.run = self._create_or_load_xml(self.xmlfilename, self.agent)

    def save(self):
#        tree = ET.ElementTree(self.root)
#        tree.write(self.xmlfilename)
        prettytree = self._prettify(self.root)
        prettytree = "\n".join([line for line in prettytree.split('\n') if line.strip() != ''])
        with open(self.xmlfilename, "w") as f:
            f.write(prettytree)

        
    def add_episode(self, new_vals, **kwargs):
        runResults = self._create_or_load_child(self.run, "runResults")
        new_vals = dict(zip(self.labels, [str(i) for i in new_vals]))
        kwargs = dict([a, str(x)] for a, x in kwargs.items())
        currep = ET.SubElement(runResults, "Episode", kwargs)
        for key, val in list(new_vals.items()):
            ET.SubElement(currep, key).text = str(val)        

#    def add_targetnetcopy(self, **kwargs):
#        kwargs = dict([a, str(x)] for a, x in kwargs.items())
#        targetnetcopys = self._create_or_load_child(self.run, "targetnetcopys")
#        ET.SubElement(targetnetcopys, "targetnetcopy", **kwargs)           

    ###############

    def _create_or_load_xml(self, filename, agent):
        if os.path.exists(filename) and ET.parse(filename).getroot().tag == "Evaluation":
            tree = ET.parse(filename)
            root = tree.getroot()
            memsize = agent.memory._size if hasattr(agent, "memory") else "NaN"
            run = ET.SubElement(root, "run", date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), memorysize=str(memsize))
            return root, run
        else:
            root = ET.Element("Evaluation", agent=agent.name)
            run = ET.SubElement(root, "run", date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            return root, run
        
    def _create_or_load_child(self, mother, childname):
        return mother.find(childname) if mother.find(childname) is not None else ET.SubElement(mother, childname)
            

    def _prettify(self, elem):
        """Return a pretty-printed XML string for the Element."""
        rough_string = ET.tostring(elem, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")

   
###############################################################################



       
        
class plotter(): 
    def __init__(self, agentname, labels, val_bounds):
        self.title = "Evaluation "+agentname
        self.labels = labels
        self.all_vals = [None]*len(labels) #is a list of lists [[],[],[]]
        self.episode = 0        
        plt.ion() 
        
        maxvals = [0]*len(val_bounds)
        minvals = [0]*len(val_bounds)
        for i in range(len(val_bounds)):
            if hasattr(val_bounds[i], "__len__"):
                minvals[i] = val_bounds[i][0]
                maxvals[i] = val_bounds[i][1]
            else:
                maxvals[i] = val_bounds[i]
        
                
        if not SHOW_IN_SUBPLOTS:
            self.figs, self.ax = plt.subplots(1,1)
            self.maxval = max(maxvals)
            self.minval = min(minvals)
        else:
            x = ceil(sqrt(len(labels)))
            y = ceil(len(labels)/x)
            self.figs, self.ax = plt.subplots(x,y)
            self.ax = flatten2(self.ax)
            self.maxvals = maxvals
            self.minvals = minvals
        self.num_epis = 100
#        self.colors = ['C%i'%i for i in range(len(labels))] 
        self.colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k','b','g','r']


    def update(self, *new_vals):
        for i in range(len(self.all_vals)):
            if self.all_vals[i] is not None:
                self.all_vals[i].append(new_vals[i])
            else:
                self.all_vals[i] = [new_vals[i]]
        self.episode += 1
        self.num_epis = self.episode if self.episode > self.num_epis else self.num_epis
        
        
        if not run_from_ipython():
            if SHOW_IN_SUBPLOTS:
                [plt.sca(i) for i in self.ax]
            else:
                plt.sca(self.ax)
        
        if SHOW_IN_SUBPLOTS:
#                [i.cla() for i in self.ax]   
            for i in range(len(self.all_vals)):
                self.ax[i].plot(range(self.episode), self.all_vals[i], self.colors[i])
                self.ax[i].axis([0, self.num_epis, self.minvals[i], self.maxvals[i]])
                self.ax[i].set_xlabel("Epoch")
                self.ax[i].xaxis.set_label_coords(0.5, 0.125)
                self.ax[i].set_ylabel(self.labels[i])
                self.ax[i].yaxis.set_label_coords(0.08, 0.5)
        else:
            self.ax.cla()
            for i in range(len(self.all_vals)):
                self.ax.plot(range(self.episode), self.all_vals[i], self.colors[i], label=self.labels[i])
            self.ax.axis([0, self.num_epis, self.minval, self.maxval])
            self.ax.legend()
            plt.xlabel('Epoch')
            
        plt.suptitle(self.title, fontsize=16)
           
        self.figs.canvas.draw()       
        plt.show()




if __name__ == '__main__':  
    ITERATIONS = 100    
    plot = plotter("name", ["val1", "val2", "val3", "val4"], [1, 1, 1, 10])
    i = 0
    while i < ITERATIONS+10:
        plot.update(i/ITERATIONS, np.random.random(), 0.5, i/(ITERATIONS+11-i))
        i+=1