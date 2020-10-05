#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 23:24:13 2020

@author: yanlan
"""

import pickle
import os
import numpy as np
import pandas as pd
import warnings; warnings.simplefilter("ignore")
import sys; sys.path.append("../Utilities/")
from newfun import get_var_bounds
from newfun import GetTrace0
import matplotlib.pyplot as plt

# parentpath = '/scratch/users/yanlan/'
# arrayid = int(os.environ['SLURM_ARRAY_TASK_ID']) # 0-935
nsites_per_id = 1000
# 
parentpath = '/Volumes/ELEMENTS/VOD_hydraulics/'
# arrayid = 10#4672
# nsites_per_id = 5
# #warmup, nsample,thinning = (0.8,2,40)


numofchains = 4
versionpath = parentpath + 'Global_0817/'
inpath = parentpath+ 'Input_Global/'
outpath = versionpath +'Output/'
cpath = versionpath+'CVG/'

MODE = 'VOD_SM_ET'
varnames, bounds = get_var_bounds(MODE)
SiteInfo = pd.read_csv('SiteInfo_globe_full.csv')

arrayid = 0
estname = cpath+'CVG_'+str(arrayid).zfill(3)+'.pkl'
with open(estname, 'rb') as f: CVG = pickle.load(f)
for arrayid in range(2,94):
    estname = cpath+'CVG_'+str(arrayid).zfill(3)+'.pkl'
    with open(estname, 'rb') as f: cvg0 = pickle.load(f)
    if cvg0.ndim==1:
        cvg = np.array(cvg0[0])
        for i in range(1,cvg0.shape[0]):
            if len(cvg0)!=18: cvg0[i] = [np.nan for ii in range(18)]
            cvg = np.row_stack([cvg,np.array(cvg0[i])])
    else:
        cvg = np.copy(cvg0)
        
    CVG = np.concatenate([CVG,cvg],axis=0)

#%%
GW,GR,CT = (CVG[:,:7],CVG[:,7:14],CVG[:,14:])

#%% 
# plt.hist(np.percentile(GR,50,axis=1),bins=np.arange(1,10,1))

# plt.hist(np.nanpercentile(GW,50,axis=1),bins=np.arange(0,1.5,.1))
# print(np.sum(np.percentile(GW,99,axis=1)<0.5)/sum(~np.isnan(np.sum(GW,axis=1))))
# print(np.sum(np.percentile(GR,50,axis=1)<2)/sum(~np.isnan(np.sum(GR,axis=1))))
# plt.hist(np.sum(np.isnan(CT),axis=1),bins=np.arange(4))
print(np.sum(np.percentile(GW,75,axis=1)<0.3)/sum(~np.isnan(np.sum(GW,axis=1))))
# print(np.sum(np.percentile(GW,25,axis=1)<0.3)/sum(~np.isnan(np.sum(GW,axis=1))))
print(np.sum(np.percentile(GW,25,axis=1)<0.3)/sum(~np.isnan(np.sum(GW,axis=1))))
# GW1 = []
# GW2 = []
# for i in range(GW.shape[0]):
#     tmp = GW[i,:][~np.isnan(GW[i,:])]
#     if len(tmp)>0:
#         GW1.append(np.percentile(tmp,75))
#         GW2.append(np.percentile(tmp,25))
#     else:
#         GW1.append(np.nan)
        # GW2.append(np.nan)
#%%
# print(np.sum(np.array(GW1)<0.3)/len(GW1))
# plt.hist(GW2)

#GW1 = [np.percentile(GW[i,:][~np.isnan(GW[i,:])],75) for i in range(GW.shape[0])]
#%%
tmpfilter = ((np.percentile(GW,25,axis=1)>0.3))
SiteInfo_p = SiteInfo.iloc[:len(tmpfilter)].iloc[tmpfilter].reset_index()

SiteInfo_p.to_csv('SiteInfo_p.csv')
# r = SiteInfo['row'][:GW.shape[0]].iloc[np.percentile(GW,25,axis=1)>0.3]
# c = SiteInfo['col'][:GW.shape[0]].iloc[np.percentile(GW,25,axis=1)>0.3]
# plt.hist(np.sum(np.isnan(CT[tmpfilter,:]),axis=1))

