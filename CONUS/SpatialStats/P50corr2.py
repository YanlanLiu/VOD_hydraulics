#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 10:03:48 2020

@author: yanlan
"""


import os
import numpy as np
import pandas as pd
import glob
import pickle
import time
import sys; sys.path.append("../Utilities/")
from newfun import LoadEnsemble
from newfun import varnames

parentpath = '/scratch/users/yanlan/' 
arrayid = int(os.environ['SLURM_ARRAY_TASK_ID']) # 0-119
nsites_per_id = 100


#parentpath = '/Volumes/ELEMENTS/VOD_hydraulics/'
#arrayid = 0
#nsites_per_id = 10

versionpath = parentpath + 'Retrieval_0510/'

outpath = versionpath+'Output/'
forwardpath = versionpath+'Forward/'
traitpath = versionpath+'Tradeoff/'
SiteInfo = pd.read_csv('../Utilities/SiteInfo_US_full.csv')
MODE = 'AM_PM_ET_'


P50corr = np.zeros([nsites_per_id,len(varnames)+3])+np.nan


#%%
tic = time.perf_counter()
for i,fid in enumerate(range(arrayid*nsites_per_id,(arrayid+1)*nsites_per_id)):
    sitename = str(SiteInfo['row'][fid])+'_'+str(SiteInfo['col'][fid])
    sitename = str(SiteInfo['row'][fid])+'_'+str(SiteInfo['col'][fid])
    paras =  LoadEnsemble(forwardpath,outpath,MODE,sitename)
    if len(paras)>0:
        P50corr[i,:] = np.corrcoef(np.transpose(paras.dropna()))[2,:]
   
toc = time.perf_counter()
print(f"Running time (1000 sites): {toc-tic:0.4f} seconds")

#%%
traitname = traitpath+'P50corr_'+str(arrayid)+'_1E3.pkl'
with open(traitname, 'wb') as f: 
    pickle.dump(P50corr, f)
