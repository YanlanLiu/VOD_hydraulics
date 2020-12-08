#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 17:57:14 2020

@author: yanlan
"""

import numpy as np
import pickle
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from newfun import GetTrace
parentpath = '/Volumes/ELEMENTS/VOD_hydraulics/'
versionpath = parentpath + 'Global_0817/'
outpath = versionpath+'Output/'
MODE = 'VOD_SM_ET'
sitename = '130_346'#'91_201'
# *_91_201*

PREFIX = outpath+MODE+'_'+sitename+'_'
trace=GetTrace(PREFIX,0,0)

# plt.plot(trace['loglik']); plt.ylim([400,800])
# plt.yscale('log')
# flist = glob(outpath+MODE+sitename+"*.pickle")
plt.plot(trace['psi50X']); #plt.ylim([400,800])
