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
IGBPlist = ['NA','ENF','EBF','DNF','DBF','DBF','Shrubland','Shrubland',
            'Savannas','Savannas','Grassland','Wetland','Cropland','Urban','Cropland','Snow']

IGBPnames = np.array([IGBPlist[itm] for itm in SiteInfo['IGBP'].values])
IGBPunique = np.unique(IGBPnames)

Nsites = 50

def getP50(TAG,varname):
    P50list = []
    outpath = parentpath + 'TroubleShooting/'+TAG+'/Output/'
    
    for fid in range(Nsites):
        sitename = str(SiteInfo['row'][fid])+'_'+str(SiteInfo['col'][fid])
        PREFIX = outpath+MODE+'_'+sitename+'_'
        trace = GetTrace(PREFIX,0,optimal=False)
        trace = trace[trace['step']>len(trace)*0.8]
        P50list.append(trace[varname].values)
    return P50list

def getR2(TAG):
    R2 = []
    statspath = parentpath + 'TroubleShooting/'+TAG+'/STATS/'
    for fid in range(Nsites):
        sitename = str(SiteInfo['row'][fid])+'_'+str(SiteInfo['col'][fid])
        with open(statspath+'R2_'+sitename+'.pkl', 'rb') as f: 
            accen, r2_vod,r2_et,r2_sm,p50_pct, Geweke = pickle.load(f)
        R2.append([np.nanmax(r2_vod),np.nanmax(r2_et),np.nanmax(r2_sm),np.nanmedian(Geweke)])
    return np.array(R2)
        
#%%

TAGlist = ['MCMC1','MC_SM2','MC_Slope2','MC_ISO','MC_ETwd']

P50list = [getP50(tag,'psi50X') for tag in TAGlist]
R2list = [getR2(tag) for tag in TAGlist]
G1list = [getP50(tag,'g1') for tag in TAGlist]



#%%
R2list = [getR2(tag) for tag in TAGlist]
dd = 0.1

for fid in [0,5,32,36,39]:#range(0,10):
    plt.figure(figsize=(6,8))
    plt.subplots_adjust(hspace=0.3)
    plt.subplot(211)
    for i in range(len(TAGlist)):
        sns.distplot(P50list[i][fid],hist=False,label=TAGlist[i])
    plt.xlim([0,15])
    plt.ylim([0,2])
    plt.ylabel('pdf')
    plt.xlabel(f'P50, ID={fid:1d}, '+str(IGBPnames[fid]))
    plt.subplot(212)
    for i in range(len(TAGlist)):
        plt.bar((i-1.5)*dd+np.arange(4),R2list[i][fid,:],width=dd)
    plt.ylim([0,1])
    
    plt.xticks(np.arange(4),['VOD','ET','SM','Geweke'])
    plt.ylabel('R2 or Geweke')
    # for i in range(len(TAGlist)):
    #     plt.bar(i+dd*np.array([-1,0,1,2]),R2list[i][fid,:],width=dd)
    # plt.ylim([-1,1])
    # plt.xticks(np.arange(len(TAGlist)),TAGlist)
    # plt.ylabel('R2 or Geweke')
