# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 11:40:12 2017

@author: csten_000
"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import matplotlib

GENAU = 400
#
#font = {'family' : 'normal',
#        'size'   : 14}
#
#matplotlib.rc('font', **font)

def main():
    fig = plt.figure(figsize=plt.figaspect(2.))
            
    x,y = [],[]
    CenterDist = np.linspace(0, 1) #0.5 ist mitte
    wallhitPunish = 1
    for i in CenterDist:
        dist = i-0.5  #abs davon ist 0 in der mitte, 0.15 vor dem curb, 0.25 mittig auf curb, 0.5 rand
        stay_on_street = ((0.5-abs(dist))*2)+0.35 #jetzt ist größer 1 auf der street
        stay_on_street = stay_on_street**0.1 if stay_on_street > 1 else stay_on_street**2 #ON street not steep, OFF street very steep 
        stay_on_street = ((1-((0.5-abs(dist))*2))**10) * -wallhitPunish + (1-(1-((0.5-abs(dist))*2))**10) *  stay_on_street #the influence of wallhitpunish is exponentially more relevant the closer to the wall you are
        stay_on_street -= 0.5        
        x.append(i)
        y.append(stay_on_street)
    ax = fig.add_subplot(2, 3, 1)
    ax.set_xlabel('Traverse Position',fontsize=14)
    ax.yaxis.set_label_coords(0.05, 0.5)
    ax.xaxis.set_label_coords(0.5, 0.05)
    ax.set_ylabel('Street-Reward',fontsize=14)
    plots = ax.plot(x,y)

    x,y = [],[]
#    angle = np.linspace(0, 1) #0.5 ist mitte
    angle = (np.linspace(-100,100))
    for i in angle:
        ang = (i  + 180 ) / 360 - 0.5
        direction_bonus = abs((0.5-(abs(ang)))*2/0.75) 
        direction_bonus = ((direction_bonus**0.4 if direction_bonus > 1 else direction_bonus**2) / 1.1 / 2) - 0.25 #no big difference until 45degrees, then BIG diff.
        x.append(i)                              
        y.append(direction_bonus)
    ax = fig.add_subplot(2, 3, 2)
    ax.set_xlabel('Car-Angle',fontsize=14)
    ax.yaxis.set_label_coords(0.05, 0.5)
    ax.xaxis.set_label_coords(0.5, 0.05)
    ax.set_ylabel('Direction-Reward',fontsize=14)
    plots += ax.plot(x,y)

    x,y = [],[]
    speed = np.linspace(0, 180) 
    for i in speed:
        sp = i/250
        tooslow = 1- ((min(0.2, sp) / 0.2) ** 3)
        x.append(i)                              
        y.append(-tooslow) 
    ax = fig.add_subplot(2, 3, 3)     
    ax.set_xlabel('Speed in longitudinal direction',fontsize=14)
    ax.yaxis.set_label_coords(0.05, 0.5)
    ax.xaxis.set_label_coords(0.5, 0.05)
    ax.set_ylim(-1, 0.1)
    ax.set_ylabel('Minimum-Speed-Reward',fontsize=14)         
    plots += ax.plot(x,y)
    
    
    walldist = np.linspace(0, 300, GENAU)
    speetinstreetdir = np.linspace(0, 180, GENAU)
    X, Y = np.meshgrid(walldist, speetinstreetdir)
    ii, jj = -1, -1
    Z = np.zeros([len(X),len(Y)])
    for i in walldist:
        ii += 1
        for j in speetinstreetdir:
            speed = j/250
            dist = i/300
            jj += 1
            speedInRelationToWallDist = dist-speed+(80/250)
            speedInRelationToWallDist = 1-(abs(speedInRelationToWallDist)*3) if speedInRelationToWallDist < 0 else (1-speedInRelationToWallDist)+0.33
            speedInRelationToWallDist = min(1,speedInRelationToWallDist)                                          
            speedInRelationToWallDist += 0.3*speed
            speedInRelationToWallDist = max(0, speedInRelationToWallDist)
            Z[jj][ii] = speedInRelationToWallDist                     
        jj = -1
    ax = fig.add_subplot(2, 3, 4, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True)
#    fig.colorbar(surf, shrink=0.5, aspect=5)    
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_xlabel('Wall-Distance',fontsize=14)
    ax.set_ylabel('Longitudinal speed',fontsize=14)
    ax.set_zlabel('Speed-Reward',fontsize=14)

    
    
    steerrange = np.linspace(-1, 1, GENAU)
    angle = np.linspace(-90, 90, GENAU)
    X, Y = np.meshgrid(steerrange, angle)
    ii, jj = -1, -1
    Z = np.zeros([len(X),len(Y)])
    for i in steerrange:
        ii += 1
        for j in angle:
            jj += 1     
            ang = j/360
            steer_bonus1 = i/5 + ang #this one rewards sterering into street-direction if the cars angle is off...
            steer_bonus1 = 0 if np.sign(steer_bonus1) != np.sign(ang) and abs(ang) > 0.15 else steer_bonus1
            steer_bonus1 = (0.5-abs(ang)) * (1-abs(steer_bonus1))
            Z[ii][jj] = steer_bonus1                     
        jj = -1
    ax = fig.add_subplot(2, 3, 5, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True)
#    fig.colorbar(surf, shrink=0.5, aspect=5)    
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_xlabel('Steering-Command',fontsize=14)
    ax.set_ylabel('Car-Angle',fontsize=14)
    ax.set_zlabel('preliminary Steer-reward',fontsize=14)    



    steerBon = np.linspace(0, 0.5, GENAU)
    CenterDist = np.linspace(0, 1, GENAU)
    X, Y = np.meshgrid(steerBon, CenterDist)
    ii, jj = -1, -1
    Z = np.zeros([len(X),len(Y)])
    for i in CenterDist:
        ii += 1
        for j in steerBon:
            jj += 1
            dist = i-0.5  #abs davon ist 0 in der mitte, 0.15 vor dem curb, 0.25 mittig auf curb, 0.5 rand
            steer_bonus1 = (abs(dist*2)) * j + (1-abs(dist*2))*0.5  #more relevant the further off you are.
            steer_bonus2 = (1-((0.5-abs(dist))*2))**10 * -abs(((i+np.sign(dist))*np.sign(dist)))/1.5   #more relevant the furhter off, steering away from wall is as valuable as doing nothing in center, doing nothing is worse, steering towards sucks 
            bon = max(0, steer_bonus1 + steer_bonus2)
            Z[ii][jj] = bon                     
        jj = -1
    ax = fig.add_subplot(2, 3, 6, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True)
#    fig.colorbar(surf, shrink=0.5, aspect=5)    
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_xlabel('preliminary Steer-reward',fontsize=14)
    ax.set_ylabel('Traverse Position',fontsize=14)
    ax.set_zlabel('Steer-reward',fontsize=14)    

#    plt.suptitle(agentname, fontsize=20, y=0.991)
#    plt.tight_layout()

#    plt.tight_layout(w_pad=-7, h_pad=-.35)

    plt.show()
    figs.set_size_inches(8, 2.5)

    fig_ext = '.png'
    figs.savefig(os.path.join('figure' + fig_ext),bbox_inches='tight', pad_inches=0)

    

if __name__ == '__main__':  
    main()


####################################################################################################################################################################################