#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 22:25:39 2020

@author: yanlan
"""

import os
import numpy as np
import pandas as pd
import sys; sys.path.append("../Utilities/")
import glob
from newfun import GetTrace
import pickle
import time

# parentpath = '/scratch/users/yanlan/' 
# arrayid = int(os.environ['SLURM_ARRAY_TASK_ID']) # 0-119
# nsites_per_id = 1000


parentpath = '/Volumes/ELEMENTS/VOD_hydraulics/'
arrayid = 1
nsites_per_id = 1

versionpath = parentpath + 'SM_0717/'
outpath = versionpath+'Output/'
traitpath = versionpath+'Traits/'
varlist = ['g1','lpx','psi50X','gpmax','C','bexp','bc']

SiteInfo = pd.read_csv('../Utilities/SiteInfo_US_full.csv')
MODE = 'AM_PM_ET_'

MissingList = []
Val_25 = np.zeros([nsites_per_id,len(varlist)])+np.nan
Val_50 = np.zeros([nsites_per_id,len(varlist)])+np.nan
Val_75 = np.zeros([nsites_per_id,len(varlist)])+np.nan
Geweke = np.zeros([nsites_per_id,])+np.nan

#%%
sample_length = int(3e3); step = int(5e2)
tic = time.perf_counter()
for i,fid in enumerate(range(arrayid*nsites_per_id,(arrayid+1)*nsites_per_id)):
    sitename = str(SiteInfo['row'][fid])+'_'+str(SiteInfo['col'][fid])
    PREFIX = outpath+MODE+sitename+'_'
    flist = glob.glob(PREFIX+'*.pickle')
    if len(flist)<5: 
        MissingList.append(fid)
    else:
        trace_df = GetTrace(PREFIX,0,optimal=False)
        var_df = trace_df[trace_df['step']>trace_df['step'].max()*0.7]
        var_df['lpx'] = var_df['lpx']*var_df['psi50X']
        Val_25[i,:] = np.array([var_df[itm].quantile(.25) for itm in varlist])
        Val_50[i,:] = np.array([var_df[itm].quantile(.50) for itm in varlist])
        Val_75[i,:] = np.array([var_df[itm].quantile(.75) for itm in varlist])
        
        st_list = range(1000,int(len(var_df)/sample_length)*sample_length-sample_length+1,step)
        gw = []
        chain = np.array(var_df['psi50X'])[:int(len(var_df)/sample_length)*sample_length] 
        tmpe = chain[-sample_length:]
        for st in st_list:
            tmps = chain[st:(st+sample_length)]
            gw.append((np.nanmean(tmps)-np.nanmean(tmpe))/np.sqrt(np.nanvar(tmps)+np.nanvar(tmpe)))
        Geweke[i] = np.nanmean(np.abs(np.array(gw)))


        
        # st_list = range(1000,int(len(var_df)/sample_length)*sample_length-sample_length+1,step)
        # gw = []
        # chainid = 0
        # for varname in varnames[:-1]:
        #     chain = np.array(var_df[varname][var_df['chain']==chainid])
        #     chain = chain[:int(len(chain)/sample_length)*sample_length] 
        #     for st in st_list:
        #         tmps = chain[st:(st+sample_length)]
        #         tmpe = chain[-sample_length:]
        #         gw.append((np.nanmean(tmps)-np.nanmean(tmpe))/np.sqrt(np.nanvar(tmps)+np.nanvar(tmpe)))
        # Geweke[i] = np.nanmean(np.abs(np.array(gw)))
        
toc = time.perf_counter()
print(f"Running time (1000 sites): {toc-tic:0.4f} seconds")

#%%
traitname = traitpath+'Traits_'+str(arrayid)+'_1E3.pkl'
with open(traitname, 'wb') as f: 
    pickle.dump([Val_25, Val_50, Val_75, Geweke,MissingList], f)
