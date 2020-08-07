#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 10:47:08 2020

@author: yanlan
"""

import os
import numpy as np
import pandas as pd
import glob
import sys; sys.path.append("../Utilities/")
from newfun import readCLM
from Utilities import MovAvg
import pickle
import time


# parentpath = '/scratch/users/yanlan/' 
# arrayid = int(os.environ['SLURM_ARRAY_TASK_ID']) # 0-14
# nsites_per_id = 1000


parentpath = '/Volumes/ELEMENTS/VOD_hydraulics/'
arrayid = 83
nsites_per_id = 1

inpath = parentpath+'Input/'
statspath = inpath+'Stats/'
SiteInfo = pd.read_csv('../Utilities/SiteInfo_US_full.csv')

mLAI = np.zeros([nsites_per_id,])+np.nan
mVOD = np.zeros([nsites_per_id,])+np.nan
mET = np.zeros([nsites_per_id,])+np.nan
N_VOD = np.zeros([nsites_per_id,])+np.nan
N_ET = np.zeros([nsites_per_id,])+np.nan
#%%
tic = time.perf_counter()
for i,fid in enumerate(range(arrayid*nsites_per_id,(arrayid+1)*nsites_per_id)):
    sitename = str(SiteInfo['row'][fid])+'_'+str(SiteInfo['col'][fid])
    Forcings,VOD,ET,dLAI,discard_vod,discard_et,idx = readCLM(inpath,sitename)
    mLAI[i] = np.nanmean(dLAI)
    mVOD[i] = np.nanmean(VOD)
    mET[i] = np.nanmean(ET)
    N_VOD[i] = np.sum(~np.isnan(VOD))
    N_ET[i] = np.sum(~np.isnan(ET))
toc = time.perf_counter()
print(f"Running time (1000 sites): {toc-tic:0.4f} seconds")

#%%
statname = statspath+'Avg'+str(arrayid)+'_1E3.pkl'
with open(statname, 'wb') as f: 
    pickle.dump([mLAI,mVOD,mET,N_VOD,N_ET], f)
