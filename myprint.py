# -*- coding: utf-8 -*-
"""
Created on Fri May 19 18:40:44 2017

@author: nivradmin
"""
#10 is highest, 1 is lowest, 5 prints everything not specified

PRINTLEVEL = 5
MAX_NORMAL_LEVEL = 10

#nicht auf 10 reduzieren, sondern sobald mal einer mit über 10 kam, printet er von da an nur noch den

def myprint(*args, **kwargs):
    global PRINTLEVEL
    try:
        level = kwargs["level"]
        if level > MAX_NORMAL_LEVEL:
           PRINTLEVEL = level
    except KeyError:
        level = 5
    if level >= PRINTLEVEL:
        print(*args)