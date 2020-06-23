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


parentpath = '/scratch/users/yanlan/' 
arrayid = int(os.environ['SLURM_ARRAY_TASK_ID']) # 0-14
nsites_per_id = 1000


# parentpath = '/Volumes/ELEMENTS/VOD_hydraulics/'
# arrayid = 83
# nsites_per_id = 1

inpath = parentpath+'Input/'
statspath = inpath+'Stats/'
SiteInfo = pd.read_csv('../Utilities/SiteInfo_US_full.csv')
varname = 'LAI'
VAR = np.zeros([nsites_per_id,])+np.nan
ValidN = VAR.copy()
#%%
tic = time.perf_counter()
for i,fid in enumerate(range(arrayid*nsites_per_id,(arrayid+1)*nsites_per_id)):
    sitename = str(SiteInfo['row'][fid])+'_'+str(SiteInfo['col'][fid])
    Forcings,VOD,ET,dLAI,discard_vod,discard_et,idx = readCLM(inpath,sitename)
    VAR[i] = np.nanmean(dLAI)
toc = time.perf_counter()
print(f"Running time (1000 sites): {toc-tic:0.4f} seconds")

#%%
r2name = statspath+varname+str(arrayid)+'_1E3.pkl'
with open(r2name, 'wb') as f: 
    pickle.dump(VAR, f)
