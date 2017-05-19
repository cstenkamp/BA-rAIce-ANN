# -*- coding: utf-8 -*-
"""
Created on Fri May 19 18:40:44 2017

@author: nivradmin
"""


PRINTLEVEL = 6

def myprint(*args, **kwargs):
    try:
        level = kwargs["level"]
        level = 10 if level > 10 else level
    except KeyError:
        level = 5
    if level >= PRINTLEVEL:
        print(*args)