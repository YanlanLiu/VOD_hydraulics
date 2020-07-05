#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 22:16:11 2020

@author: yanlan
"""

from random import randint
import pickle
import os
import numpy as np
import pandas as pd
import warnings; warnings.simplefilter("ignore")
import sys; sys.path.append("../Utilities/")
from newfun import readCLM # newfun_full
from newfun import fitVOD_RMSE,dt, hour2day, hour2week
from newfun import OB,CONST,CLAPP,ca,varnames
from newfun import GetTrace
from Utilities import MovAvg, nanOLS
import time
import matplotlib.pyplot as plt

# =========================== control pannel =============================

# parentpath = '/scratch/users/yanlan/'
# arrayid = int(os.environ['SLURM_ARRAY_TASK_ID']) # 0-119
# nsites_per_id = 1
# warmup, nsample,thinning = (0.7,200,20)

parentpath = '/Volumes/ELEMENTS/VOD_hydraulics/'

TAG = 'MCMC1'
versionpath = parentpath + 'TroubleShooting/'+TAG+'/'
outpath = versionpath +'Output/'

MODE = 'AM_PM_ET'
SiteInfo = pd.read_csv('SiteInfo_reps_50.csv')

for fid in range(10,12):

    sitename = str(SiteInfo['row'][fid])+'_'+str(SiteInfo['col'][fid])
    PREFIX = outpath+MODE+'_'+sitename+'_'
    trace = GetTrace(PREFIX,0,optimal=False)
    plt.figure()
    plt.plot(trace['psi50X'])
    plt.xlabel(TAG+', '+str(fid))
# for v in varnames:
#     plt.figure()
#     plt.plot(trace[v])
#     plt.ylabel(v)