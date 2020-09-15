#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 11:10:06 2020

@author: yanlan
"""

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="ticks", color_codes=True,font_scale=1.75)
import numpy as np
from Utilities import dailyAvg
versionpath = '/Volumes/ELEMENTS/VOD_hydraulics/Global_0817/Pixels/'
inpath = versionpath+'Input/'
forwardpath = versionpath+'Forward/'
SiteInfo = pd.read_csv('pixels.csv')


def augTS(TS,discard):
    TS_full = np.zeros(discard.shape)+np.nan
    TS_full[~discard] = TS
    return TS_full

for fid in range(10,21):
    sitename = str(SiteInfo['row'].values[fid])+'_'+str(SiteInfo['col'].values[fid])
    with open(inpath+'Input_'+sitename+'.pkl','rb') as f: 
        Forcings,VOD,SOILM,ET,dLAI,discard_vod,discard_et,idx = pickle.load(f)
    with open(forwardpath+'TS_VOD_SM_ET_'+sitename+'.pkl','rb') as f:
        TS = pickle.load(f)
    meanTS = [np.nanmean(itm,axis=0) for itm in TS]
    VOD_hat,ET_hat,PSIL_hat,S1_hat = meanTS
    
    RNET,TEMP,P,VPD,Psurf,GA,LAI,VegK = Forcings

    xlim = [150,912]
    # xlim=[650,800]
    plt.figure(figsize=(10,8))
    plt.subplot(311)
    plt.plot(augTS(SOILM,discard_vod[1::2]),label='AMSRE')
    plt.plot(augTS(S1_hat,discard_vod[1::2]),label='Assimilated')
    plt.legend()
    plt.xlim(xlim)
    plt.ylabel('Soil moisture')
    plt.subplot(312)
    dP = dailyAvg(P,8)
    plt.plot(dP)
    plt.ylabel('Precipitation')
    plt.xlim(xlim)
    plt.subplot(313)
    dT = dailyAvg(TEMP-273,8)
    plt.plot(dT)
    plt.ylabel('Temperature')
    plt.xlabel('Day, '+sitename)
    plt.xlim(xlim)

#%%
