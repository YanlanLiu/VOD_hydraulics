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
from newfun import OB,CONST,CLAPP,ca
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

TAG = 'MC_SM2'
versionpath = parentpath + 'TroubleShooting/'+TAG+'/'

# TAG = 'MC_SM'
# versionpath = parentpath + TAG; 

outpath = versionpath +'Output/'

MODE = 'AM_PM_ET'
IGBPlist = ['NA','ENF','EBF','DNF','DBF','DBF','Shrubland','Shrubland',
            'Savannas','Savannas','Grassland','Wetland','Cropland','Urban','Cropland','Snow','NA','NA']

SiteInfo = pd.read_csv('SiteInfo_reps_50.csv')
# SiteInfo = pd.read_csv('../Utilities/SiteInfo_US_full.csv')
IGBP = [IGBPlist[itm] for itm in SiteInfo['IGBP'].values]
varnames  = ['g1','lpx','psi50X','gpmax','C','bexp','bc','sigma_et','sigma_vod','sigma_sm','loglik']

#%%
SMsigma = []
for fid in range(0,50):

    sitename = str(SiteInfo['row'][fid])+'_'+str(SiteInfo['col'][fid])
    PREFIX = outpath+MODE+'_'+sitename+'_'
    trace = GetTrace(PREFIX,varnames,0,optimal=False)
    trace = trace[trace['step']>trace['step'].max()*0.8].reset_index().drop(columns=['index'])
    SMsigma.append(trace['sigma_sm'].mean())
#%%
# np.median(SMsigma)
# np.percentile(SMsigma,25)
# np.percentile(SMsigma,75)

#%%
    # plt.figure()
    # plt.plot(trace['psi50X'])
    # # print(len(trace))
    # plt.xlabel(TAG+', '+str(fid)+', '+IGBP[fid])
# for v in varnames:
#     plt.figure()
#     plt.plot(trace[v])
#     plt.ylabel(v)