#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 15:30:05 2020

@author: yanlan

To what extent VOD is affected by leaf water potential
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="ticks", color_codes=True,font_scale=1.5)
import warnings; warnings.simplefilter("ignore")
import sys; sys.path.append("../Utilities/")
from newfun_full import readCLM
from Utilities import MovAvg, nancoef

parentpath = '/Volumes/ELEMENTS/VOD_hydraulics/'
TAG = 'WT_ISO'
versionpath = parentpath + 'TroubleShooting/'+TAG+'/'
inpath = parentpath+ 'Input/'

forwardpath = versionpath+'Forward/'
statspath = versionpath+'STATS/'

MODE = 'AM_PM_ET'
SiteInfo = pd.read_csv('SiteInfo_reps.csv')

R2 = np.zeros([100,2])
for fid in range(100):
    print(fid)
    sitename = str(SiteInfo['row'][fid])+'_'+str(SiteInfo['col'][fid])

    Forcings,VOD,SOILM,ET,dLAI,discard_vod,discard_et,idx = readCLM(inpath,sitename)
    VOD_ma = np.reshape(VOD,[-1,2])
    VOD_ma = np.reshape(np.column_stack([MovAvg(VOD_ma[:,0],4),MovAvg(VOD_ma[:,1],4)]),[-1,])
    
    
    with open(statspath+'R2_'+sitename+'.pkl', 'rb') as f: 
        enacc,r2_vod,r2_et,r2_sm,p50_pct,Geweke = pickle.load(f)
    
    R2[fid,:] = np.array([nancoef(VOD_ma,dLAI)**2,np.nanmax(r2_vod)])

#%%
plt.figure(figsize=(5,5))
plt.plot(R2[:,1],R2[:,0],'ob')
plt.plot([0,1],[0,1],'--k')
plt.xlabel(r'R$^2$ of VOD = $f(\psi_l, LAI)$')
plt.ylabel(r'R$^2$ of VOD = $f(\overline{\psi_l}, LAI)$')
plt.figure()
plt.hist(R2[:,1]-R2[:,0],bins=np.arange(0,1,0.05),density=True)
plt.xlabel(r'% of VOD variance explained by $\psi_l$')