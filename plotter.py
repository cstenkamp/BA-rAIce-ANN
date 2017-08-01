import matplotlib.pyplot as plt
import numpy as np

class plotter():
    def __init__(self, planned_num_epis, title, *labels):        
        plt.ion() 
        self.figs, self.ax = plt.subplots(1,1)
        self.num_epis = planned_num_epis
        self.title = title
        self.labels = labels
        self.all_vals = [None]*len(labels) #is a list of lists [[],[],[]]
        self.episode = 0
        self.colors = ['C%i'%i for i in range(len(labels))]
        
    
    def update(self, *new_vals):
        for i in range(len(self.all_vals)):
            if self.all_vals[i] is not None:
                self.all_vals[i].append(new_vals[i])
            else:
                self.all_vals[i] = [new_vals[i]]
        self.episode += 1
        self.num_epis = self.episode if self.episode > self.num_epis else self.num_epis
        
        if not self.run_from_ipython:
            plt.sca(self.ax)
        self.ax.cla()
        for i in range(len(self.all_vals)):
            plt.plot(range(self.episode), self.all_vals[i], self.colors[i], label=self.labels[i])
    
        plt.axis([0, self.num_epis, 0, 1])
        self.ax.legend()
        plt.title(self.title)
        plt.xlabel('Epoch')   
        self.figs.canvas.draw()       
        plt.show()
        plt.pause(0.0001) 
    
    
    def run_from_ipython(self):
        try:
            __IPYTHON__
            return True
        except NameError:
            return False
    
ITERATIONS = 100    
plot = plotter(ITERATIONS, "Thetitle", "val1", "val2")
i = 0
while i < ITERATIONS+10:
    plot.update(i/ITERATIONS, np.random.random())
    i+=1
     
    
############################### WORKING SCRIPT ################################
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
#    plt.plot(x,y, 'C1', label='C1')
#    plt.plot(x,z, 'C2', label='C2')
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
    
             
    
