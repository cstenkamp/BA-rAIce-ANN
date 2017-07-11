# -*- coding: utf-8 -*-
"""
Created on Fri May 19 18:40:44 2017

@author: nivradmin
"""
#10 is highest, 1 is lowest, 5 prints everything not specified

PRINTLEVEL = 5

def myprint(*args, **kwargs):
    try:
        level = kwargs["level"]
        level = 10 if level > 10 else level
    except KeyError:
        level = 5
    if level >= PRINTLEVEL:
        print(*args)