# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 11:40:12 2017

@author: csten_000
"""
import sys
import xml.etree.ElementTree as ET
import config
import os
import numpy as np
from math import sqrt, ceil
import matplotlib.pyplot as plt

flatten = lambda l: [item for sublist in l for item in sublist]
def flatten2(l):
    try:
        return flatten(l)
    except:
        return l

averageForPrint = 1
epsiloncolor = "0.5"
prettystrings = {"maxq": "best average Q", "maxrew": "best average reward", "avgtime": "average laptime", "maxprog": "best progress"} if averageForPrint != 1 else {"maxq": "average Q", "maxrew": "average reward", "avgtime": "laptime", "maxprog": "progress"}
                
MINMAXOVERWRITE = {"maxprog":[-8, 100], "avgtime":[0,60]}
MAXEPIPRINT = 9999

def prettify(string):
    if string in prettystrings:
        return prettystrings[string]
    else:
        return string
    

def main(agentname, nonrl=False):
    global print_epsilon_in
    conf = config.Config()
    filename = os.path.join(conf.superfolder(), agentname, conf.xml_dir, agentname+"_eval.xml")
    allruns = read_xml(filename, nonrl)
    toplot = average_and_extract(allruns, nonrl)
    minmax = extract_minmax(toplot, MINMAXOVERWRITE)
#    print(toplot)
#    print([i["epsilon"] for i in toplot])
#    labels = list(minmax.keys())
    print_epsilon_in = [] if nonrl else ["maxrew"]
    labels = ["maxprog", "avgtime"] if nonrl else ["maxprog", "maxrew", "avgtime", "maxq"] 
    plot(agentname, labels, minmax, toplot, nonrl)

    

def extract_minmax(ls, overwrites):
    aslist = [list(i.values()) for i in ls]
    perval = list(zip(*aslist))
    minmax = [[np.min(i),np.max(i)] for i in perval]
    tmp = dict(zip(*[list(ls[0].keys()), minmax]))
    tmp = {**tmp, **overwrites}
    return tmp



def plot(agentname, labels, val_bounds, all_vals, nonrl=False):
    
    aslist = list(zip(*[list(i.values()) for i in all_vals]))
    aslist = dict(zip(*[list(all_vals[0].keys()), aslist]))
#    print(aslist["num"])
    rng = range(val_bounds["num"][1])
    rng = [averageForPrint* i for i in rng]
    
    maxval = list(rng)[-1]
    
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k','b','g','r']

    x = ceil(sqrt(len(labels)))
    y = ceil(len(labels)/x)
    figs, ax = plt.subplots(x,y)
    ax = flatten2(ax)
    
    
    plt.tight_layout()
    
    
    j = 0
    for i in labels:
                
        plots = ax[j].plot(rng,aslist[i],colors[j], label=prettify(i))
        
        if i in print_epsilon_in:
            ax2 = ax[j].twinx()
            plots = plots + ax2.plot(rng,aslist["epsilon"], epsiloncolor, label="epsilon")
            ax2.axis([0, maxval, 0, 1])
            ax2.yaxis.set_label_coords(0.96, 0.5)
            ax2.set_ylabel('epsilon', color=epsiloncolor)
            ax2.tick_params('y', colors=epsiloncolor)
        
        ax[j].axis([0, maxval, val_bounds[i][0], val_bounds[i][1]])
        ax[j].set_xlabel("Epoch", fontsize=15)
        ax[j].xaxis.set_label_coords(0.5, 0.06)
        
        if not nonrl:
            iters = aslist["iteration"][-1]
            ax3 = ax[j].twiny()
            ax3.set_xlabel("iteration")
            ax3.xaxis.set_label_coords(0.5, 0.94)
            ax3.set_xlim(0, iters)
            steps = [0.1, 0.4, 0.6, 0.9]
            ax3.set_xticks([int(i*iters) for i in steps])
            ax3.set_xticklabels([str(int(i*iters)) for i in steps])
        
        
        ax[j].set_ylabel(i, fontsize=12)
        ax[j].yaxis.set_label_coords(0.05, 0.5)
        print([l.get_label() for l in plots])
        ax[j].legend(plots, [l.get_label() for l in plots], fontsize=11, loc=4)
        
        j += 1
    
    plt.suptitle(agentname, fontsize=20, y=0.991)
    plt.show()
    figs.set_size_inches(8, 2.5)

    fig_ext = '.png'
    figs.savefig(os.path.join('figure' + fig_ext),bbox_inches='tight', pad_inches=0)
    



def average_and_extract(runs, nonrl=False):
    onlyimportant = []
    for i in runs:
        if nonrl:
            tmp = {"progress": round(float(i["progress"]),2),
                   "laptime": round(float(i["laptime"]),2)}
        else:
            tmp = {"iteration": int(i["endIteration"]), 
                   "netsteps": int(i["reinfNetSteps"]), 
                   "progress": round(float(i["progress"]),2),
                   "laptime": round(float(i["laptime"]),2),
                   "epsilon": round(float(i["endEpsilon"]),5),
                   "Qvals": round(float(i["average_Q-vals"]),2),
                   "rewards": round(float(i["average_rewards"]),2)}
        onlyimportant.append(tmp)
    
    
    averaged = []
    ind = 0
    print(len(onlyimportant), "episodes")
    for i in range(-1, min(len(onlyimportant),MAXEPIPRINT), averageForPrint):
        ind += 1
        maxval = min(len(onlyimportant),i+averageForPrint)
#        print([onlyimportant[j]["progress"] for j in  range(i,maxval)])
        tmp = {}
        tmp["maxprog"] = np.max([onlyimportant[j]["progress"] for j in  range(i,maxval)])
        tmp["avgtime"] = np.mean([onlyimportant[j]["laptime"] for j in  range(i,maxval)])
        tmp["num"] = ind
        if not nonrl:
            tmp["step"] = onlyimportant[maxval-1]["netsteps"]
            tmp["iteration"] = onlyimportant[maxval-1]["iteration"]
            tmp["maxq"] = np.max([onlyimportant[j]["Qvals"] for j in  range(i,maxval)])
            tmp["maxrew"] = np.max([onlyimportant[j]["rewards"] for j in  range(i,maxval)])
            tmp["epsilon"] = np.mean([onlyimportant[j]["epsilon"] for j in  range(i,maxval)])
        averaged.append(tmp)
    
    return averaged
    

def read_xml(FileName, nonrl=False):
    tree = ET.parse(FileName)
    root = tree.getroot()
    assert root.tag=="Evaluation", "that is not the kind of XML I thought it would be."
    allruns = []
    for majorpoint in root:
        if majorpoint.tag == "run":
            currun = []
            for minorpoint in majorpoint:
                if minorpoint.tag == "runResults":
                    for currpoint in minorpoint:
                        tmp = currpoint.attrib
                        for stuff in currpoint:
                            tmp[stuff.tag] = stuff.text
                        currun.append(tmp)
            allruns.append(currun)
       
    if not nonrl:
        #sometimes, the last values of a run are not saved and need to be removed
    #    print([len(i) for i in allruns])
        for i in range(1, len(allruns)):
            maxi = allruns[i][0]["startMemoryEntry"]
            for j in range(len(allruns[i-1])-1,-1,-1):
                if allruns[i-1][j]["startMemoryEntry"] < maxi:
                    break
            allruns[i-1] = allruns[i-1][:j]
        
    allruns = [x for x in allruns if x != []]
    allruns = flatten2(allruns)
    
    return allruns

    

if __name__ == '__main__':  
    if "--agent" in sys.argv:
        num = sys.argv.index("--agent")
        try:
            agentname = sys.argv[num+1]
            if agentname[0] == "-": raise IndexError
        except IndexError:
            print("With the '--agent'-Parameter, you need to specify an agent!")
            exit(0)
    else:
        if "-svplay" in sys.argv:
            agentname = config.Config().standardSVAgent
        else:
            agentname = config.Config().standardAgent
    
    main(agentname, ("-nonrl" in sys.argv))