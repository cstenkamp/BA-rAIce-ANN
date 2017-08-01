import matplotlib.pyplot as plt
import numpy as np
import threading 
from utils import run_from_ipython
from math import sqrt, ceil
flatten = lambda l: [item for sublist in l for item in sublist]

class plotter():
    def __init__(self, show_plot, planned_num_epis, title, show_in_subplots, labels, maxvals):
        self.title = title
        self.labels = labels
        self.all_vals = [None]*len(labels) #is a list of lists [[],[],[]]
        self.episode = 0
        self.show_plot = show_plot
        if self.show_plot:
            plt.ion() 
            if not show_in_subplots:
                self.figs, self.ax = plt.subplots(1,1)
                self.maxval = max(maxvals)
            else:
                x = ceil(sqrt(len(labels)))
                y = ceil(len(labels)/x)
                self.figs, self.ax = plt.subplots(x,y)
                self.ax = flatten(self.ax)
                self.maxvals = maxvals
            self.show_in_subplots = show_in_subplots
            self.num_epis = planned_num_epis
            self.colors = ['C%i'%i for i in range(len(labels))]
        
    
    def _update(self, *new_vals):
        for i in range(len(self.all_vals)):
            if self.all_vals[i] is not None:
                self.all_vals[i].append(new_vals[i])
            else:
                self.all_vals[i] = [new_vals[i]]
        self.episode += 1
        self.num_epis = self.episode if self.episode > self.num_epis else self.num_epis
        
        if self.show_plot:
            try:
                if not run_from_ipython():
                    if self.show_in_subplots:
                        [plt.sca(i) for i in self.ax]
                    else:
                        plt.sca(self.ax)
                
                if self.show_in_subplots:
                    [i.cla() for i in self.ax]   
                    for i in range(len(self.all_vals)):
                        self.ax[i].plot(range(self.episode), self.all_vals[i], self.colors[i])
                        self.ax[i].axis([0, self.num_epis, 0, self.maxvals[i]])
                        self.ax[i].set_xlabel("Epoch")
                        self.ax[i].xaxis.set_label_coords(0.5, 0.125)
                        self.ax[i].set_ylabel(self.labels[i])
                        self.ax[i].yaxis.set_label_coords(0.08, 0.5)
                else:
                    self.ax.cla()
                    for i in range(len(self.all_vals)):
                        self.ax.plot(range(self.episode), self.all_vals[i], self.colors[i], label=self.labels[i])
                    self.ax.axis([0, self.num_epis, 0, self.maxval])
                    self.ax.legend()
                    plt.xlabel('Epoch')
                    
                plt.suptitle(self.title, fontsize=16)
                   
                self.figs.canvas.draw()       
                plt.show()
                plt.pause(0.00001) 
            except ValueError: #dann wurde er geschlossen
                pass
    

    def update(self, *new_vals):
        t1 = threading.Thread(target=self._update, args=(new_vals))
        t1.start()








if __name__ == '__main__':  
    ITERATIONS = 100    
    plot = plotter(True, ITERATIONS, "Thetitle", True, ["val1", "val2", "val3", "val4"], [1, 1, 1, 10])
    i = 0
    while i < ITERATIONS+10:
        plot.update(i/ITERATIONS, np.random.random(), 0.5, i/(ITERATIONS+11-i))
        i+=1
     
    
###################### WORKING SCRIPT (multiple subplots) ####################
#plt.ion() 
#figs, ax = plt.subplots(1,2)     
#
#i=0
#x=list()
#y=list()
#z=list()
#
#ITERATIONS = 100
#
#while i <ITERATIONS:
#    temp_y=np.random.random();
#    x.append(i);
#    y.append(temp_y);
#    z.append(i/ITERATIONS)
##    plt.scatter(i,temp_y);
#               
#    #plt.sca([i for i in ax])
#    [i.cla() for i in ax]
#    ax[0].plot(x,y, 'C1', label='C1')
#    ax[1].plot(x,z, 'C2', label='C2')
#    [i.axis([0, ITERATIONS-1, 0, 1]) for i in ax]
#    plt.title('Training...')
#    plt.ylabel('Accuracy')
#    plt.xlabel('Epoch')   
#    [i.legend() for i in ax]
#    figs.canvas.draw()         
#        
#    i+=1;
#    plt.show()
#    plt.pause(0.0001) 





###################### WORKING SCRIPT (single plot) ###########################
#plt.ion() 
#figs, ax = plt.subplots(1,1)     
#
#i=0
#x=list()
#y=list()
#z=list()
#
#ITERATIONS = 100
#
#while i <ITERATIONS:
#    temp_y=np.random.random();
#    x.append(i);
#    y.append(temp_y);
#    z.append(i/ITERATIONS)
##    plt.scatter(i,temp_y);
#               
#    plt.sca(ax)
#    ax.cla()     
#    ax.plot(x,y, 'C1', label='C1')
#    ax.plot(x,z, 'C2', label='C2')
#    plt.axis([0, ITERATIONS-1, 0, 1])
#    plt.title('Training...')
#    plt.ylabel('Accuracy')
#    plt.xlabel('Epoch')   
#    ax.legend()
#    figs.canvas.draw()         
#        
#    i+=1;
#    plt.show()
#    plt.pause(0.0001) 