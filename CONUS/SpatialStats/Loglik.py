#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 14:35:51 2020

@author: yanlan
"""

import os
import numpy as np
import pandas as pd
import glob
from newfun import GetTrace
import pickle
import time
import sys; sys.path.append("../Utilities/")

# parentpath = '/scratch/users/yanlan/' 
# arrayid = int(os.environ['SLURM_ARRAY_TASK_ID']) # 0-119
# nsites_per_id = 1000


parentpath = '/Volumes/ELEMENTS/VOD_hydraulics/'
arrayid = 0
nsites_per_id = 10

versionpath = parentpath + 'Retrieval_0510/'
outpath = versionpath+'Output/'
likpath = versionpath+'Loglik/'
varlist = ['sigma_et','sigma_vod','loglik']

SiteInfo = pd.read_csv('../Utilities/SiteInfo_US_full.csv')
MODE = 'AM_PM_ET_'

MissingList = []
Val_25 = np.zeros([nsites_per_id,len(varlist)])+np.nan
Val_50 = np.zeros([nsites_per_id,len(varlist)])+np.nan
Val_75 = np.zeros([nsites_per_id,len(varlist)])+np.nan

#%%
tic = time.perf_counter()
for i,fid in enumerate(range(arrayid*nsites_per_id,(arrayid+1)*nsites_per_id)):
    sitename = str(SiteInfo['row'][fid])+'_'+str(SiteInfo['col'][fid])
    PREFIX = outpath+MODE+sitename+'_'
    flist = glob.glob(PREFIX+'*.pickle')
    if len(flist)<5: 
        MissingList.append(fid)
    else:
        trace_df = GetTrace(PREFIX,0,optimal=False)
        var_df = trace_df[trace_df['step']>trace_df['step'].max()*0.8]
        Val_25[i,:] = np.array([var_df[itm].quantile(.25) for itm in varlist])
        Val_50[i,:] = np.array([var_df[itm].quantile(.50) for itm in varlist])
        Val_75[i,:] = np.array([var_df[itm].quantile(.75) for itm in varlist])
        
toc = time.perf_counter()
print(f"Running time (1000 sites): {toc-tic:0.4f} seconds")

#%%
traitname = likpath+'Loglik_'+str(arrayid)+'_1E3.pkl'
with open(traitname, 'wb') as f: 
    pickle.dump([Val_25, Val_50, Val_75, MissingList], f)
