#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 20:34:25 2020

Calculate GelmanRubin and Geweke from trace

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
import time

tic = time.perf_counter()

# parentpath = '/scratch/users/yanlan/'
# arrayid = int(os.environ['SLURM_ARRAY_TASK_ID']) # 0-935
# #arrayid = 849
# nsites_per_id = 100
# warmup, nsample,thinning = (0.8,20,100)
# 
parentpath = '/Volumes/ELEMENTS/VOD_hydraulics/'
arrayid = 10#4672
nsites_per_id = 100
warmup, nsample,thinning = (0.8,2,40)



cc = [0,1,2]
versionpath = parentpath + 'Global_0817/'
inpath = parentpath+ 'Input_Global/'
outpath = versionpath +'Output/'
forwardpath = versionpath+'Forward/'
statspath = versionpath+'STATS/'
cpath = versionpath+'CVG/'

MODE = 'VOD_SM_ET'
varnames, bounds = get_var_bounds(MODE)
SiteInfo = pd.read_csv('SiteInfo_globe_full.csv')


#%%
sample_length=int(5e3); step = int(1e3)

CVG = []; CVGnan = np.array([np.nan for i in range(17)])
# for fid in range(10,11):
for fid in range(arrayid*nsites_per_id,min((arrayid+1)*nsites_per_id,len(SiteInfo))):
    # fid = 10
    sitename = str(SiteInfo['row'].values[fid])+'_'+str(SiteInfo['col'].values[fid])
    print(sitename)
    flist = [outpath+MODE+'_'+sitename+'_'+str(chainid).zfill(2)+'_'+str(chunckid).zfill(2)+'.pickle' for chainid in range(3) for chunckid in range(20)]
    try:
        trace = GetTrace0(flist,0)
    except:
        print("No trace")
        CVG.append(CVGnan)
        
    trace_s = trace.sort_values(by=['loglik']).reset_index().drop(columns=['index'])
    st_list = range(0,int(len(trace_s)/sample_length)*sample_length-sample_length+1,step)



    paralist=varnames[:7]
    cc = np.unique(trace['chain'])
    GW = np.zeros([len(paralist)])
    GR = np.zeros([len(paralist),])
    M = len(cc)
    N = np.copy(sample_length)
    
    
    for j,varname in enumerate(paralist):
        st_list = range(0,int(len(trace_s)/sample_length)*sample_length-sample_length+1,step)
        chain = np.array(trace_s[varname])[-int(len(trace_s)/sample_length)*sample_length:] 
        Geweke = []
        for st in st_list:
            tmps = chain[st:(st+sample_length)]
            tmpe = chain[-sample_length:]
            Geweke.append((np.nanmean(tmps)-np.nanmean(tmpe))/np.sqrt(np.nanvar(tmps)+np.nanvar(tmpe)))
        GW[j] = np.quantile(np.abs(Geweke[int(len(st_list)*0.8):]),.1)
    
    
        st_list = range(0,int((trace['step'].max())/sample_length)*sample_length-sample_length,step)
        GelmenRubin = []
        for st in st_list:
            # subtrace = trace.sort_values(by=['loglik']).reset_index()
            tmp_trace = trace[(trace['step']>=st) & (trace['step']<st+sample_length)]
            # tmp_trace = trace_s.iloc[st:st+sample_length]
            #n = np.mean(np.array([len(trace_df[varname][trace_df['chain']==chainid]) for chainid in range(10)]))
            theta_m = np.array([tmp_trace[varname][tmp_trace['chain']==chainid].mean() for chainid in cc])
            s_m = np.array([tmp_trace[varname][tmp_trace['chain']==chainid].var() for chainid in cc])
        
            B = np.var(theta_m)*M/(M-1)*N
            W = np.mean(s_m)
            V = (N-1)/N*W+(M+1)/M/N*B
            GelmenRubin.append(np.sqrt(V/W))
        GR[j] = np.quantile(GelmenRubin[int(len(st_list)*0.5):],.2)
    
    Count = (trace_s[int(len(trace_s)*0.8):].reset_index().drop(columns=['index']))['chain'].value_counts().values
    
    entry = np.concatenate([GW,GR,Count])
    CVG.append(entry)
#%%   

CVG = np.array(CVG)
print(CVG.shape)

estname = cpath+'CVG_'+str(arrayid).zfill(3)+'.pkl'
with open(estname, 'wb') as f: 
    pickle.dump(CVG, f)

toc = time.perf_counter()
    
    
print(f"Running time (100 sites): {toc-tic:0.4f} seconds")

