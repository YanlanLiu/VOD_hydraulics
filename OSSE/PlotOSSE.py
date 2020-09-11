#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 08:02:52 2020

@author: yanlan
"""

import pickle
import os
import numpy as np
import pandas as pd
import warnings; warnings.simplefilter("ignore")
import sys; sys.path.append("../GLOBAL/")
from newfun import readCLM # newfun_full
from newfun import fitVOD_RMSE,calVOD,dt, hour2day, hour2week
from newfun import get_var_bounds,OB,CONST,CLAPP,ca
from newfun import GetTrace
from Utilities import nanOLS,nancorr,MovAvg,IsOutlier
import time
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="ticks", color_codes=True,font_scale=1.75)

# parentpath = '/scratch/users/yanlan/'
# arrayid = int(os.environ['SLURM_ARRAY_TASK_ID'])
parentpath = '/Volumes/ELEMENTS/VOD_hydraulics/'
versionpath = parentpath + 'OSSE/'

inpath = parentpath+ 'Input_Global/'
outpath = versionpath +'Output/'
datapath = versionpath+'FakeData/'
statspath = versionpath+'STATS/'

MODE = 'VOD_SM_ET'
varnames, bounds = get_var_bounds(MODE)
#SiteInfo = pd.read_csv('SiteInfo_reps_53.csv')
SiteInfo = pd.read_csv('../GLOBAL/SiteInfo_globe_full.csv')

fidlist = np.arange(1000,len(SiteInfo),1000)


#%% True values
THETA = []; tnan = [np.nan for i in range(len(varnames))]
for i in range(len(fidlist)):
    fid = fidlist[i]
    sitename = str(SiteInfo['row'].values[fid])+'_'+str(SiteInfo['col'].values[fid])
    fname = datapath+'Para_'+sitename+'.pkl'
    if os.path.isfile(fname):
        with open(fname,'rb') as f:
            theta = pickle.load(f)
        THETA.append(theta)
    else:
        THETA.append(tnan)
THETA = np.array(THETA)

THETA[:,2] = -THETA[:,2]
# THETA[:,1] = THETA[:,1]*THETA[:,2]

#%% Retrieved values
Collection_ACC = np.zeros([len(fidlist),4])+np.nan
Collection_PARA = np.zeros([len(fidlist),14])+np.nan
Collection_STD = np.zeros([len(fidlist),11])+np.nan

Collection_OBS = np.zeros([len(fidlist),9])+np.nan
Collection_N = np.zeros([len(fidlist),3])+np.nan


nsites_per_id = 5
for arrayid in range(19):
    subrange = np.arange(arrayid*nsites_per_id,min((arrayid+1)*nsites_per_id,len(fidlist)))
    fname = statspath+'EST_'+MODE+'_'+str(arrayid).zfill(3)+'.pkl'
    if os.path.isfile(fname):
        with open(fname,'rb') as f:
            TS_mean,std,PARA_mean,PARA_std,PARA2_mean,PARA2_std,ACC = pickle.load(f)
        if ACC.shape[1]>0:
            Collection_ACC[subrange,:] = ACC
            Collection_PARA[subrange,:] = PARA_mean
            Collection_STD[subrange,:] = PARA2_std
    fname = statspath+'OBS_'+MODE+'_'+str(arrayid).zfill(3)+'.pkl'
    if os.path.isfile(fname):
        with open(fname,'rb') as f:
            OBS_mean,OBS_std,OBS_N = pickle.load(f)            
        if OBS_mean.shape[1]>0:
            Collection_OBS[subrange,:] = OBS_mean
            Collection_N[subrange,:] = OBS_N

Collection_PARA[:,2] = -Collection_PARA[:,2]
# Collection_PARA[:,1] = Collection_PARA[:,1]*Collection_PARA[:,2]


#%%
Filter = ~((np.abs(Collection_PARA[:,2]-THETA[:,2])>3) | (np.abs(Collection_PARA[:,0]-THETA[:,0])>2.4) | 
           (np.abs(Collection_PARA[:,3]-THETA[:,3])>5) |
           (np.abs(Collection_PARA[:,5]-THETA[:,5])>3) | (np.isnan(np.sum(THETA,axis=1))))
print(sum(Filter))
#%%
vnamelist = [r'$g_1$',r'$\psi_{50,s}/\psi_{50,x}$',r'$\psi_{50,x}$',r'$g_{p,max}$',r'$C$',r'soil$_b$',r'soil$_{bc}$']
plt.figure(figsize=(16,7))
plt.subplots_adjust(wspace=.38,hspace=.42)
for i,vname in enumerate(vnamelist[:7]):
    plt.subplot('24'+str(i+1))
    yhat = Collection_PARA[Filter,i]; y = THETA[Filter,i]
    std = Collection_STD[Filter,i]
    # plt.figure(figsize=(4,4))
    dd = (max(y)-min(y))*0.1
    ylim = [min(y)-dd,max(y)+dd]
    plt.plot(y,yhat,'ok')
    plt.plot(ylim,ylim,'--k')
    plt.xlim(ylim);plt.ylim(ylim)
    for j in range(len(y)):
        plt.plot(y[j]*np.array([1,1]),yhat[j]+std[j]*np.array([-1,1]),'-',color='grey')
    r = np.corrcoef(y,yhat)[0][1]
    plt.title(vname+f", r = {r:.2f}")
    
    if i==6:
        j = 1
        plt.plot(y[j],yhat[j],'ok',label ='Mean')
        plt.plot(y[j]*np.array([1,1]),yhat[j]+std[j]*np.array([-1,1]),'-',color='grey',label=r'$\pm$std.')
        plt.legend(loc=0,bbox_to_anchor=(2.1,0.75))
        plt.text(-0.65,-0.45,'True traits',fontsize=24,weight='bold')
    if i==4:
        plt.text(-12,20,'Estimated traits',fontsize=24,weight='bold',rotation=90)


#%%

