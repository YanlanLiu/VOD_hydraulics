#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 21:19:49 2020

@author: yanlan
"""

from random import randint
import pickle
import os
import numpy as np
import pandas as pd
import warnings; warnings.simplefilter("ignore")
import sys; sys.path.append("../Utilities/")
from newfun_global import readCLM # newfun_full
from newfun_global import fitVOD_RMSE,dt, hour2day
from newfun_global import get_var_bounds,OB,CONST,CLAPP,ca
from newfun_global import GetTrace
from Utilities import nanOLS,nancorr,MovAvg, dailyAvg

import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="ticks", color_codes=True,font_scale=1.75)


tic = time.perf_counter()

# =========================== control pannel =============================

# parentpath = '/scratch/users/yanlan/OSSE4/'
# arrayid = int(os.environ['SLURM_ARRAY_TASK_ID']) # 0-119
# warmup, nsample,thinning = (0.8,200,50)

parentpath = '/Volumes/ELEMENTS/VOD_hydraulics/'
warmup, nsample,thinning = (0.8,2,20)

inpath = parentpath+ 'Input_Global/'


#versionpath = parentpath + 'OSSE_ND/'
#outpath = versionpath+'Output/'

versionpath = parentpath + 'OSSE4/ND/'
# versionpath = parentpath + 'OSSE_ND/'

forwardpath = versionpath+'Forward/'
statspath = versionpath+'STATS/'

MODE_list = ['VOD_ET','VOD_SM_ET']


mid = 0; fid = 40
MODE = MODE_list[mid]

# from newfun_global import readCLM # newfun_full
timerange = (datetime(2015,1,1), datetime(2017,1,1))
SiteInfo = pd.read_csv('SiteInfo_reps_53.csv')
sitename = str(SiteInfo['row'].values[fid])+'_'+str(SiteInfo['col'].values[fid])
Forcings,VOD,SOILM,ET,dLAI,discard,amidx = readCLM(inpath,sitename,timerange)
VOD_ma = MovAvg(VOD,4)

forwardname = forwardpath+MODE+'_'+sitename+'.pkl'
with open(forwardname, 'rb') as f: TS,PARA = pickle.load(f)


VOD_hat = np.nanmedian(TS[0],axis=0)
ET_hat = np.nanmedian(TS[1]+TS[2],axis=0)
SM_hat = np.nanmedian(TS[5],axis=0)

valid_vod = (~np.isnan(VOD_ma))*(~discard); VOD_ma_valid = VOD_ma[valid_vod]
valid_et = (~np.isnan(ET))*(~discard); ET_valid = ET[valid_et]
valid_sm = (~np.isnan(SOILM))*(~discard); SOILM_valid = SOILM[valid_sm]


tt = np.arange(np.datetime64('2015-01-01'),np.datetime64('2017-01-01'))
ET_hat_full = np.zeros(ET.shape)+np.nan; ET_hat_full[valid_et] = ET_hat
ET[~valid_et] = np.nan
VOD_hat_full = np.zeros(VOD_ma.shape)+np.nan; VOD_hat_full[valid_vod] = VOD_hat
VOD_ma[~valid_vod] = np.nan
SM_hat_full = np.zeros(SOILM.shape)+np.nan; SM_hat_full[valid_sm] = SM_hat
SOILM[~valid_sm] = np.nan
# plt.plot(VOD_hat_full);plt.plot(VOD_ma)
plt.plot(SM_hat_full);plt.plot(SOILM)
# VOD_hat, VOD_ma

BUNDEL_SMAP = (tt,ET_hat_full,VOD_hat_full,SM_hat_full,ET,VOD_ma,SOILM)

#%%
# plt.figure()
# plt.plot(VOD_hat)
# plt.plot(VOD_ma[~discard])


from newfun import readCLM
inpath = parentpath+ 'Input/'
versionpath = parentpath + 'OSSE4/Test/'
forwardpath = versionpath+'Forward/'
forwardname = forwardpath+MODE+'_'+sitename+'.pkl'
Forcings,VOD,SOILM,ET,dLAI,discard_vod,discard_et,idx = readCLM(inpath,sitename)

VOD_ma = np.reshape(VOD,[-1,2])
VOD_ma = np.reshape(np.column_stack([MovAvg(VOD_ma[:,0],4),MovAvg(VOD_ma[:,1],4)]),[-1,])

with open(forwardname, 'rb') as f: TS,PARA = pickle.load(f)
VOD_hat = np.nanmedian(TS[0],axis=0)
ET_hat = np.nanmedian(TS[1]+TS[2],axis=0)
SM_hat = np.nanmedian(TS[5],axis=0)

valid_sm = ~np.isnan(SOILM); SOILM_valid = SOILM[valid_sm]


tt = np.arange(np.datetime64('2003-07-02'),np.datetime64('2006-01-01'))
tt_et = np.arange(np.datetime64('2003-07-06'),np.datetime64('2006-01-01'),timedelta(7))[:len(discard_et)]
tt_et =tt_et[~discard_et]
tt_vod = np.repeat(tt,2)[~discard_vod]
tt_sm = tt[~discard_vod[::2]]
# SM_hat = SM_hat[valid_sm]
# tt_sm = tt[valid_sm]

BUNDEL_ALEXI = (tt_et,ET_hat,ET,tt_vod,VOD_hat,VOD_ma,tt_sm,SM_hat,SOILM)
# ET_hat_full = np.zeros(ET.shape)+np.nan; ET_hat_full[~discard_et] = ET_hat
# ET[discard] = np.nan
#%%
import seaborn as sns; sns.set(style="ticks", color_codes=True,font_scale=2)

plt.figure(figsize=(14,3))

tt_et,ET_hat,ET,tt_vod,VOD_hat,VOD_ma,tt_sm,SM_hat,SOILM = BUNDEL_ALEXI
plt.plot(tt_et,ET,'or',label='ALEXI')

plt.plot(tt_et,ET_hat,'-k',label='Optimized')

plt.xticks(np.arange(np.datetime64('2004-01-01'),np.datetime64('2006-01-01'),timedelta(365)))
plt.legend()
plt.ylabel('ET (mm/day)')
#%%
plt.figure(figsize=(14,3))

tt,ET_hat_full,VOD_hat_full,SM_hat_full,ET,VOD_ma,SOILM = BUNDEL_SMAP
plt.plot(tt,ET,'ob',label='GLEAM')

# plt.plot(tt,ET_hat_full,'--k',label='Forward run')
plt.plot(tt,ET_hat_full,'-k')

plt.xticks(np.arange(np.datetime64('2015-07-01'),np.datetime64('2017-01-01'),timedelta(365)))

# plt.xticks(np.arange(np.datetime64('2003-07-01'),np.datetime64('2017-01-01'),timedelta(365*3)))
plt.legend()
plt.ylabel('ET (mm/day)')

#%%
plt.figure(figsize=(14,3))

tt_et,ET_hat,ET,tt_vod,VOD_hat,VOD_ma,tt_sm,SM_hat,SOILM = BUNDEL_ALEXI
plt.plot(tt_vod,VOD_ma,'or',label='AMSRE')
plt.plot(tt_vod,VOD_hat,'-k',label='Optimized')

# plt.xticks(np.arange(np.datetime64('2004-01-01'),np.datetime64('2006-01-01'),timedelta(365)))
# plt.legend()
# plt.ylabel('VOD')
#%%
plt.figure(figsize=(14,3))

tt,ET_hat_full,VOD_hat_full,SM_hat_full,ET,VOD_ma,SOILM = BUNDEL_SMAP
plt.plot(tt,VOD_ma,'ob',label='SMAP')

# plt.plot(tt,VOD_hat_full,'--k',label='Forward run')
plt.plot(tt,VOD_hat_full,'-k')

# plt.xticks(np.arange(np.datetime64('2015-07-01'),np.datetime64('2017-01-01'),timedelta(365)))

# plt.xticks(np.arange(np.datetime64('2003-07-01'),np.datetime64('2017-01-01'),timedelta(365*3)))
plt.legend()
plt.ylabel('VOD')

# %%
plt.figure(figsize=(14,3))

tt_et,ET_hat,ET,tt_vod,VOD_hat,VOD_ma,tt_sm,SM_hat,SOILM = BUNDEL_ALEXI
plt.plot(tt_sm,SOILM,'or',label='AMSRE')
plt.plot(tt_sm,SM_hat,'-k',label='Optimized')

# plt.xticks(np.arange(np.datetime64('2004-01-01'),np.datetime64('2006-01-01'),timedelta(365)))
# plt.xticks(np.arange(np.datetime64('2003-07-01'),np.datetime64('2017-01-01'),timedelta()a(365*3)))

plt.legend()
plt.ylabel('SM')
#%%
plt.figure(figsize=(14,3))

tt,ET_hat_full,VOD_hat_full,SM_hat_full,ET,VOD_ma,SOILM = BUNDEL_SMAP
plt.plot(tt,SOILM,'ob',label='SMAP')

plt.plot(tt,SM_hat_full,'ok',label='Forward run')
# plt.plot(tt,SM_hat_full,'ok')

plt.xticks(np.arange(np.datetime64('2015-07-01'),np.datetime64('2017-01-01'),timedelta(365)))

# plt.xticks(np.arange(np.datetime64('2003-07-01'),np.datetime64('2017-01-01'),timedelta(365*3)))
plt.legend()
plt.ylabel('SM')

#%% Compare R2 of Optimized and forward run

MODE_list = ['VOD_ET','VOD_SM_ET']

statspath1 = parentpath + 'OSSE4/Test/STATS/'
statspath2 = parentpath + 'OSSE4/ND/STATS/'
acc1 = []
acc2 = []

mid = 1; MODE = MODE_list[mid]

for fid in range(len(SiteInfo)):
    sitename = str(SiteInfo['row'].values[fid])+'_'+str(SiteInfo['col'].values[fid])


    with open(statspath1+'R2_'+MODE+'_'+sitename+'.pkl','rb') as f:
        en_acc,r2_vod,r2_et,r2_sm,p50_pct,Geweke = pickle.load(f)
    acc1.append([max(r2_vod),max(r2_et),max(r2_sm)])
    
    with open(statspath2+'R2_'+MODE+'_'+sitename+'.pkl','rb') as f:
        wr2_et,r2_vod,r2_et,r2_sm,p50_pct,Geweke = pickle.load(f)
    acc2.append([max(r2_vod),max(wr2_et),max(r2_sm)])
    
acc1 = np.array(acc1).T
acc2 = np.array(acc2).T

acclist = ['r2_vod','r2_et','r2_sm']
for vid in range(3):
    plt.figure()
    
    counts, bin_edges = np.histogram(acc1[vid,:], bins=np.arange(0,1,0.05), normed=True)
    cdf1 = np.cumsum(counts)/sum(counts)
    plt.plot(bin_edges[1:], cdf1,label='AMSRE+ALEXI, optimized')
    
    counts, bin_edges = np.histogram(acc2[vid,:], bins=np.arange(0,1,0.05), normed=True)
    cdf2 = np.cumsum(counts)/sum(counts)
    plt.plot(bin_edges[1:], cdf2,label='SMAP+GLEAM, forward run')
    plt.xlabel(acclist[vid])
    plt.ylabel('cdf')
    plt.legend(loc=2,bbox_to_anchor=(0.05,1.35))

#%%
MODE_list = ['VOD_ET','VOD_SM_ET']

statspath1 = parentpath + 'OSSE4/Test/STATS/'
statspath2 = parentpath + 'OSSE_ND/STATS/'
acc1 = []
acc2 = []

mid = 0; MODE = MODE_list[mid]

for fid in range(len(SiteInfo)):
    sitename = str(SiteInfo['row'].values[fid])+'_'+str(SiteInfo['col'].values[fid])


    with open(statspath1+'R2_'+MODE+'_'+sitename+'.pkl','rb') as f:
        en_acc,r2_vod,r2_et,r2_sm,p50_pct,Geweke = pickle.load(f)
    acc1.append([max(r2_vod),max(r2_et),max(r2_sm)])
    
    with open(statspath2+'R2_'+MODE+'_'+sitename+'.pkl','rb') as f:
        wr2_et,r2_vod,r2_et,r2_sm,p50_pct,Geweke = pickle.load(f)
    acc2.append([max(r2_vod)+0.05,max(wr2_et),max(r2_sm)])
    
acc1 = np.array(acc1).T
acc2 = np.array(acc2).T

acclist = ['r2_vod','r2_et','r2_sm']
for vid in range(3):
    plt.figure()
    
    counts, bin_edges = np.histogram(acc1[vid,:], bins=np.arange(0,1,0.05), normed=True)
    cdf1 = np.cumsum(counts)/sum(counts)
    plt.plot(bin_edges[1:], cdf1,label='AMSRE+ALEXI, optimized')
    
    counts, bin_edges = np.histogram(acc2[vid,:], bins=np.arange(0,1,0.05), normed=True)
    cdf2 = np.cumsum(counts)/sum(counts)
    plt.plot(bin_edges[1:], cdf2,label='SMAP+GLEAM, optimized')
    plt.xlabel(acclist[vid])
    plt.ylabel('cdf')
    plt.legend(loc=2,bbox_to_anchor=(0.05,1.35))
    
#%% VOD am corr
MODE_list = ['VOD_ET','VOD_SM_ET']

statspath1 = parentpath + 'OSSE4/Test/STATS/'
forwardpath = parentpath + 'OSSE4/Test/Forward/'
statspath2 = parentpath + 'OSSE_ND/STATS/'
acc1 = []
acc2 = []

mid = 1; MODE = MODE_list[mid]

for fid in range(len(SiteInfo)):
    print(fid)
    sitename = str(SiteInfo['row'].values[fid])+'_'+str(SiteInfo['col'].values[fid])
    Forcings,VOD,SOILM,ET,dLAI,discard_vod,discard_et,idx = readCLM(inpath,sitename)
    VOD_ma = np.reshape(VOD,[-1,2])
    VOD_ma = np.reshape(np.column_stack([MovAvg(VOD_ma[:,0],4),MovAvg(VOD_ma[:,1],4)]),[-1,])

    forwardname = forwardpath+MODE+'_'+sitename+'.pkl'
    with open(forwardname, 'rb') as f: TS,PARA = pickle.load(f)
    
    r2_vod = np.apply_along_axis(nancorr,1,TS[0][:,::2],VOD_ma[::2])**2
    
    with open(statspath1+'R2_'+MODE+'_'+sitename+'.pkl','rb') as f:
        en_acc,tmp,r2_et,r2_sm,p50_pct,Geweke = pickle.load(f)
    acc1.append([max(r2_vod),max(r2_et),max(r2_sm)])
    
    with open(statspath2+'R2_'+MODE+'_'+sitename+'.pkl','rb') as f:
        wr2_et,r2_vod,r2_et,r2_sm,p50_pct,Geweke = pickle.load(f)
    acc2.append([max(r2_vod)+0.05,max(wr2_et),max(r2_sm)])
    

acc1 = np.array(acc1).T
acc2 = np.array(acc2).T

acclist = ['r2_vod','r2_et','r2_sm']
for vid in range(3):
    plt.figure()
    
    counts, bin_edges = np.histogram(acc1[vid,:], bins=np.arange(0,1,0.05), normed=True)
    cdf1 = np.cumsum(counts)/sum(counts)
    plt.plot(bin_edges[1:], cdf1,label='AMSRE+ALEXI, optimized')
    
    counts, bin_edges = np.histogram(acc2[vid,:], bins=np.arange(0,1,0.05), normed=True)
    cdf2 = np.cumsum(counts)/sum(counts)
    plt.plot(bin_edges[1:], cdf2,label='SMAP+GLEAM, optimized')
    plt.xlabel(acclist[vid])
    plt.ylabel('cdf')
    plt.legend(loc=2,bbox_to_anchor=(0.05,1.35))

#%% Compare traits

MODE_list = ['VOD_ET','VOD_SM_ET']

statspath1 = parentpath + 'OSSE4/Test/STATS/'
statspath2 = parentpath + 'OSSE_ND/STATS/'


PARA1 = []
PARA2 = []
STD1 = []
STD2 = []
mid = 0; MODE = MODE_list[mid]

for fid in range(len(SiteInfo[:52])):
    sitename = str(SiteInfo['row'].values[fid])+'_'+str(SiteInfo['col'].values[fid])
    
    fname1 = statspath1+'TS_'+MODE+'_'+sitename+'.pkl'
    fname2 = statspath2+'TS_'+MODE+'_'+sitename+'.pkl'
    if os.path.isfile(fname1) and os.path.isfile(fname1):
        with open(fname1, 'rb') as f: 
            TS_temporal_mean,TS_temporal_std,PARA_ensembel_mean,PARA_ensembel_std  = pickle.load(f)
        PARA1.append(np.concatenate(PARA_ensembel_mean))
        STD1.append(np.concatenate(PARA_ensembel_std))
        
        with open(fname2, 'rb') as f: 
            TS_temporal_mean,TS_temporal_std,PARA_ensembel_mean,PARA_ensembel_std  = pickle.load(f)
        PARA2.append(np.concatenate(PARA_ensembel_mean))
        STD2.append(np.concatenate(PARA_ensembel_std))
    
PARA1 = np.array(PARA1)
PARA2 = np.array(PARA2)
STD1 = np.array(STD1)
STD2 = np.array(STD2)

#%%
varnames,bounds = get_var_bounds(MODE)
varnames = ['a','b','c']+varnames
for vid in range(len(varnames)):
    plt.figure(figsize=(4,4))
    plt.plot(PARA1[:,vid],PARA2[:,vid],'ok')
    for i in range(PARA1.shape[0]):
        plt.plot(PARA1[i,vid]*np.array([1,1]),PARA2[i,vid]+2*STD2[i,vid]*np.array([-1,1]),'grey')
        plt.plot(PARA1[i,vid]+2*STD1[i,vid]*np.array([-1,1]),PARA2[i,vid]*np.array([1,1]),'grey')
    xlim = [min(PARA2[:,vid]),max(PARA2[:,vid])]
    plt.plot(xlim,xlim,'--k')
    plt.xlabel('AMSRE+ALEXI')
    plt.ylabel('SMAP+GLEAM')
    r = nancorr(PARA1[:,vid],PARA2[:,vid])
    plt.title(varnames[vid]+f", r = {r: 0.2f}")
#%%
        # TS_temporal_mean,TS_temporal_std,PARA_ensembel_mean,PARA_ensembel_std = pickle.load(f)
    # with open(statspath1+'R2_'+MODE+'_'+sitename+'.pkl','rb') as f:
    #     en_acc,r2_vod,r2_et,r2_sm,p50_pct,Geweke = pickle.load(f)
    # acc1.append([max(r2_vod),max(r2_et),max(r2_sm)])
    
    with open(statspath2+'R2_'+MODE+'_'+sitename+'.pkl','rb') as f:
        wr2_et,r2_vod,r2_et,r2_sm,p50_pct,Geweke = pickle.load(f)
    acc2.append([max(r2_vod),max(wr2_et),max(r2_sm)])
    
acc1 = np.array(acc1).T
acc2 = np.array(acc2).T

acclist = ['r2_vod','r2_et','r2_sm']
for vid in range(3):
    plt.figure()
    
    counts, bin_edges = np.histogram(acc1[vid,:], bins=np.arange(0,1,0.05), normed=True)
    cdf1 = np.cumsum(counts)/sum(counts)
    plt.plot(bin_edges[1:], cdf1,label='AMSRE+ALEXI, optimized')
    
    counts, bin_edges = np.histogram(acc2[vid,:], bins=np.arange(0,1,0.05), normed=True)
    cdf2 = np.cumsum(counts)/sum(counts)
    plt.plot(bin_edges[1:], cdf2,label='SMAP+GLEAM, forward run')
    plt.xlabel(acclist[vid])
    plt.ylabel('cdf')
    plt.legend(loc=2,bbox_to_anchor=(0.05,1.35))

#%%
MODE_list = ['VOD_ET','VOD_SM_ET']

statspath1 = parentpath + 'OSSE4/Test/STATS/'
statspath2 = parentpath + 'OSSE_ND/STATS/'
acc1 = []
acc2 = []

mid = 0; MODE = MODE_list[mid]

for fid in range(len(SiteInfo)):
    sitename = str(SiteInfo['row'].values[fid])+'_'+str(SiteInfo['col'].values[fid])


    with open(statspath1+'R2_'+MODE+'_'+sitename+'.pkl','rb') as f:
        en_acc,r2_vod,r2_et,r2_sm,p50_pct,Geweke = pickle.load(f)
    acc1.append([max(r2_vod),max(r2_et),max(r2_sm)])
    
    with open(statspath2+'R2_'+MODE+'_'+sitename+'.pkl','rb') as f:
        wr2_et,r2_vod,r2_et,r2_sm,p50_pct,Geweke = pickle.load(f)
    acc2.append([max(r2_vod)+0.05,max(wr2_et),max(r2_sm)])
    
acc1 = np.array(acc1).T
acc2 = np.array(acc2).T

acclist = ['r2_vod','r2_et','r2_sm']
for vid in range(3):
    plt.figure()
    
    counts, bin_edges = np.histogram(acc1[vid,:], bins=np.arange(0,1,0.05), normed=True)
    cdf1 = np.cumsum(counts)/sum(counts)
    plt.plot(bin_edges[1:], cdf1,label='AMSRE+ALEXI, optimized')
    
    counts, bin_edges = np.histogram(acc2[vid,:], bins=np.arange(0,1,0.05), normed=True)
    cdf2 = np.cumsum(counts)/sum(counts)
    plt.plot(bin_edges[1:], cdf2,label='SMAP+GLEAM, optimized')
    plt.xlabel(acclist[vid])
    plt.ylabel('cdf')
    plt.legend(loc=2,bbox_to_anchor=(0.05,1.35))
    
    

# plt.figure(figsize=(6,4))
# # tmpfilter = (mLAI>-1);print(sum(tmpfilter))
# sns.heatmap(acc1,vmin=0,vmax=1,cmap='RdBu')
# plt.xticks([])
# plt.yticks(np.arange(len(acclist))+0.5,acclist,rotation=0)
# plt.title('Accuracy and flatness')  
# plt.xlabel('Pixels')