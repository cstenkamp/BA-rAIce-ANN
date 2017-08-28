# -*- coding: utf-8 -*-
"""
Created on Fri May 19 18:40:44 2017

@author: nivradmin
"""
#10 is highest, 1 is lowest, 5 prints everything not specified

from collections import deque

PRINTLEVEL = 5
MAX_NORMAL_LEVEL = 10
ONLYONCELEVEL = -1

#nicht auf 10 reduzieren, sondern sobald mal einer mit über 10 kam, printet er von da an nur noch den

onceprintedobjects = deque(100*[None], 100)


def myprint(*args, **kwargs):
    global PRINTLEVEL
    try:
        level = kwargs["level"]
        if level > MAX_NORMAL_LEVEL:
           PRINTLEVEL = level
    except KeyError:
        level = 5
    if level == ONLYONCELEVEL:
        if not args in onceprintedobjects:
            onceprintedobjects.append(args)
            level = PRINTLEVEL
    if level >= PRINTLEVEL:
        print(*args)
       
        
        
        
        
def printtofile(*args, **kwargs):
    global PRINTLEVEL
    try:
        level = kwargs["level"]
        if level > MAX_NORMAL_LEVEL:
           PRINTLEVEL = level
    except KeyError:
        level = 5
    if level == ONLYONCELEVEL:
        if not args in onceprintedobjects:
            onceprintedobjects.append(args)
            level = PRINTLEVEL
    if level >= PRINTLEVEL:
        with open("log.txt", "a") as myfile:
            args = [str(i) for i in args]
            text = " ".join(args)
            print(text)
            myfile.write(text+"\n")