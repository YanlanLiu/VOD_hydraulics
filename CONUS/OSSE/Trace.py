#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 12:34:05 2020

@author: yanlan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 22:16:11 2020

@author: yanlan
"""

from random import randint
import pickle
import os
import numpy as np
import pandas as pd
import warnings; warnings.simplefilter("ignore")
import sys; sys.path.append("../Utilities/")
from newfun import readCLM # newfun_full
from newfun import fitVOD_RMSE,dt, hour2day, hour2week
from newfun import OB,CONST,CLAPP,ca
from newfun import GetTrace, get_var_bounds
from Utilities import MovAvg, nanOLS
import time
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="ticks", color_codes=True,font_scale=1.75)

# =========================== control pannel =============================

parentpath = '/Volumes/ELEMENTS/VOD_hydraulics/'
datapath = parentpath + 'OSSE/FakeData/'

TAG = 'High'; noise_level = 2
versionpath = parentpath + 'OSSE/'+TAG+'/'; 
outpath = versionpath +'Output/'

IGBPlist = ['NA','ENF','EBF','DNF','DBF','DBF','Shrubland','Shrubland',
            'Savannas','Savannas','Grassland','Wetland','Cropland','Urban','Grassland','Snow','NA','NA']

SiteInfo = pd.read_csv('SiteInfo_reps_55.csv')
IGBP = [IGBPlist[itm] for itm in SiteInfo['IGBP'].values]

MODE_list = ['VOD_ET','VOD_ET_ISO','VOD_SM','VOD_SM_ISO','VOD_SM_ET','VOD_SM_ET_ISO']

mid = 0
MODE = MODE_list[mid]
varnames, bounds = get_var_bounds(MODE)

#%%
SMsigma = []
for fid in range(5,6):
    
    sitename = str(SiteInfo['row'][fid])+'_'+str(SiteInfo['col'][fid])
    
    with open(datapath+'Para_'+sitename+'.pkl', 'rb') as f: theta,popt = pickle.load(f)
    
    PREFIX = outpath+MODE+'_'+sitename+'_'
    trace = GetTrace(PREFIX,varnames,0,optimal=False)
    # plt.figure()
    # plt.plot(trace['psi50X'])
    # plt.plot([0,len(trace)],theta[2]*np.array([1,1]))
    # plt.xlabel(MODE+','+TAG+', '+str(fid)+', '+IGBP[fid])
    plt.figure()
    sns.kdeplot(trace['psi50X'][trace['step']>50000]) 
    ylim = plt.gca().get_ylim()
    plt.plot(theta[2]*np.array([1,1]),ylim)

#%%
def getP50(MODE):
    varnames, bounds = get_var_bounds(MODE)
    P50 = []
    for fid in range(55):
        sitename = str(SiteInfo['row'][fid])+'_'+str(SiteInfo['col'][fid])
        PREFIX = outpath+MODE+'_'+sitename+'_'
        trace = GetTrace(PREFIX,varnames,0,optimal=False)
        P50.append(trace['psi50X'][trace['step']>50000].values)
    return P50
P50 = [getP50(MODE) for MODE in MODE_list]
#%%

P50_true = []
for fid in range(55):
    sitename = str(SiteInfo['row'][fid])+'_'+str(SiteInfo['col'][fid])
    with open(datapath+'Para_'+sitename+'.pkl', 'rb') as f: theta,popt = pickle.load(f)
    P50_true.append(theta[2])

with open(TAG+'.pkl', 'wb') as f: pickle.dump((P50,P50_true),f)
#%%
TAG = 'High'
with open(TAG+'.pkl','rb') as f: 
    P50, P50_true = pickle.load(f)
c = ['b','r','g','purple','orange','navy']
for fid in range(50,55):
    plt.figure()
    for mid in range(len(MODE_list)):
        if np.std(P50[mid][fid])>0.1 :
            sns.kdeplot(P50[mid][fid],label=MODE_list[mid],color=c[mid])
    ylim = np.array(plt.gca().get_ylim())
    ylim[1] = min(ylim[1],5)
    plt.plot(P50_true[fid]*np.array([1,1]),ylim,'-k')
    plt.xlim([0,12])
    plt.ylim(ylim)
    plt.xlabel('High'+', '+str(fid)+', '+IGBP[fid])
    plt.legend(bbox_to_anchor=(1.75,1.05))
    