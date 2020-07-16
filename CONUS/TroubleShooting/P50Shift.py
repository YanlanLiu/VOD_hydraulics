#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 11:57:18 2020

@author: yanlan

Compare P50 shifts of different cost functions
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns; sns.set(style="ticks", color_codes=True,font_scale=1.5)
import sys; sys.path.append("../Utilities/")
from newfun import GetTrace

parentpath = '/Volumes/ELEMENTS/VOD_hydraulics/'
MODE = 'AM_PM_ET'
SiteInfo = pd.read_csv('SiteInfo_reps_50.csv')
Nsites = 50

def getP50(TAG):
    P50list = []
    outpath = parentpath + 'TroubleShooting/'+TAG+'/Output/'
    
    for fid in range(Nsites):
        sitename = str(SiteInfo['row'][fid])+'_'+str(SiteInfo['col'][fid])
        PREFIX = outpath+MODE+'_'+sitename+'_'
        trace = GetTrace(PREFIX,30000,optimal=False)
        P50list.append(trace['psi50X'].values)
    return P50list


P50_MC = getP50('MCMC1')
P50_SM = getP50('MC_SM')
# P50_Slope=getP50('MC_Slope')
P50_ISO=getP50('MC_ISO')
P50_ETwd=getP50('MC_ETwd')

#%% Accuracy

def getR2(TAG):
    R2 = []
    statspath = parentpath + 'TroubleShooting/'+TAG+'/STATS/'
    for fid in range(Nsites):
        sitename = str(SiteInfo['row'][fid])+'_'+str(SiteInfo['col'][fid])
        with open(statspath+'R2_'+sitename+'.pkl', 'rb') as f: 
            accen, r2_vod,r2_et,r2_sm,p50_pct, Geweke = pickle.load(f)
        R2.append([np.nanmax(r2_vod),np.nanmax(r2_et)])
    return np.array(R2)

R2_MC = getR2('MCMC1')
# R2_SM = getR2('MC_SM')
R2_ISO = getR2('MC_ISO')
R2_ETwd = getR2('MC_ETwd')


#%%
for fid in range(1):
    plt.figure(figsize=(6,8))
    plt.subplot(211)
    sns.distplot(P50_MC[fid],hist=False,label='MC')
    # sns.distplot(P50_SM[fid],hist=False,label='SM')
    sns.distplot(P50_ISO[fid],hist=False,label='MC_ISO')
    sns.distplot(P50_ETwd[fid],hist=False,label='MC_ETwd')
    # plt.xlabel('ID = '+str(fid))
    plt.xlim([0,15])
    plt.subplot(212)
    plt.bar(np.arange(3),[R2_MC[fid,0],R2_ISO[fid,0],R2_ETwd[fid,0]],width=0.3)
    plt.bar(np.arange(3)+0.3,[R2_MC[fid,1],R2_ISO[fid,1],R2_ETwd[fid,1]],width=0.3)
