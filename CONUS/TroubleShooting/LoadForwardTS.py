#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 22:07:19 2020

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
import matplotlib.pyplot as plt
# parentpath = '/scratch/users/yanlan/' 
# arrayid = int(os.environ['SLURM_ARRAY_TASK_ID']) # 0-14
nsites_per_id = 1000


parentpath = '/Volumes/ELEMENTS/VOD_hydraulics/'
#arrayid = 83
#nsites_per_id = 1

versionpath = parentpath + 'Retrieval_0510/'
inpath = parentpath+'Input/'
outpath = versionpath+'Output/'
#forwardpath = parentpath+'Forward_test/'
#r2path = parentpath+'R2_test/'
forwardpath = versionpath+'Forward/'
r2path = versionpath+'R2/'

SiteInfo = pd.read_csv('../Utilities/SiteInfo_US_full.csv')
MODE = 'AM_PM_ET_'

trb = pd.read_csv('trb_list.csv')


#%%
tid = 30
tmp = trb.iloc[tid]
fid = np.where((SiteInfo['row']==tmp['row']) & (SiteInfo['col']==tmp['col']))[0][0]
arrayid = int(fid/nsites_per_id)
sitename = str(SiteInfo['row'][fid])+'_'+str(SiteInfo['col'][fid])
Forcings,VOD,ET,dLAI,discard_vod,discard_et,idx = readCLM(inpath,sitename)

VOD_ma = np.reshape(VOD,[-1,2])
VOD_ma = np.reshape(np.column_stack([MovAvg(VOD_ma[:,0],4),MovAvg(VOD_ma[:,1],4)]),[-1,])
# plt.plot(VOD_ma)
forwardname = forwardpath+MODE+sitename+'.pkl'
if os.path.isfile(forwardname):
    with open(forwardname,'rb') as f:  # Python 3: open(..., 'rb')
        SVOD, SET, SPSIL,SPOPT = pickle.load(f)

plt.figure()
plt.plot(ET,'-r')
plt.plot(np.nanmean(SET,axis=0),'-',color='navy')

plt.figure()
plt.plot(VOD,'-r')
plt.plot(np.nanmean(SVOD,axis=0),'-',color='navy')

# Next, calculate logliklihood of ET and VOD. See if VOD smashes ET. If so, play weith traits

# plt.plot(np.nanmin(SVOD,axis=0),'--',color='lightblue')
