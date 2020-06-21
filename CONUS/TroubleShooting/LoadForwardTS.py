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
from scipy.stats import norm
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
traitpath = versionpath+'Traits/'
likpath = versionpath+'Loglik/'


SiteInfo = pd.read_csv('../Utilities/SiteInfo_US_full.csv')
MODE = 'AM_PM_ET_'

trb = pd.read_csv('trb_list.csv')

A0 = SiteInfo[['row','col']]
A1 = trb[['row','col']]
A2 = A1.merge(A0,how='outer',indicator=True).loc[lambda x : x['_merge']=='right_only']
 
#%%
tid = 1000
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
plt.subplot(211)
plt.plot(ET,'-r')
plt.plot(np.nanmean(SET,axis=0),'-',color='navy')
plt.subplot(212)
plt.plot(VOD,'-r')
plt.plot(np.nanmean(SVOD,axis=0),'-',color='navy')

# Next, calculate logliklihood of ET and VOD. See if VOD smashes ET. If so, play weith traits

likname = likpath+'Loglik_'+str(arrayid)+'_1E3.pkl'

with open(likname, 'rb') as f: 
    Lik_25, Lik_50, Lik_75, MissingList = pickle.load(f)
     
sigma_et, sigma_vod, loglik = Lik_25[fid-arrayid*nsites_per_id,:]
valid_vod = ~np.isnan(VOD_ma)
valid_et = ~np.isnan(ET)
loglik_vod = np.nanmean(norm.logpdf(VOD_ma,np.nanmean(SVOD,axis=0),sigma_vod))
loglik_et = np.nanmean(norm.logpdf(ET,np.nanmean(SET,axis=0),sigma_et))

print([loglik_vod,loglik_et])

