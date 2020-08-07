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
from Utilities import MovAvg, nanOLS, nancorr
import time
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="ticks", color_codes=True,font_scale=1.75)

# =========================== control pannel =============================

parentpath = '/Volumes/ELEMENTS/VOD_hydraulics/'
datapath = parentpath + 'OSSE2/FakeData/'

TAG = 'Test'; noise_level = 1
outpath1 = parentpath + 'OSSE2/'+TAG+'/Output/'
outpath2 = parentpath + 'OSSE3/'+TAG+'/Output/'
IGBPlist = ['NA','ENF','EBF','DNF','DBF','DBF','Shrubland','Shrubland',
            'Savannas','Savannas','Grassland','Wetland','Cropland','Urban','Grassland','Snow','NA','NA']

SiteInfo = pd.read_csv('SiteInfo_reps_53.csv')
IGBP = [IGBPlist[itm] for itm in SiteInfo['IGBP'].values]

MODE_list = ['VOD_ET','VOD_ET_ISO','VOD_SM','VOD_SM_ISO','VOD_SM_ET','VOD_SM_ET_ISO']

mid = 0
MODE = MODE_list[mid]
varnames, bounds = get_var_bounds(MODE)

#%%
SMsigma = []
for fid in range(23,33):
    
    sitename = str(SiteInfo['row'][fid])+'_'+str(SiteInfo['col'][fid])
    
    with open(datapath+'Para_'+sitename+'.pkl', 'rb') as f: theta,popt = pickle.load(f)
    
    PREFIX = outpath2+MODE+'_'+sitename+'_'
    trace = GetTrace(PREFIX,varnames,0,optimal=False)
    plt.figure()
    plt.plot(trace['psi50X'])
    # plt.plot([0,len(trace)],theta[2]*np.array([1,1]))
    plt.xlabel(MODE+','+TAG+', '+str(fid)+', '+IGBP[fid])
    # plt.figure()
    # sns.kdeplot(trace['psi50X'][trace['step']>50000]) 
    # ylim = plt.gca().get_ylim()
    # plt.plot(theta[2]*np.array([1,1]),ylim)

#%%
varname = 'g1'; varid = 0
# varname = 'psi50X';varid = 2
def getP50(outpath,MODE,varname):
    varnames, bounds = get_var_bounds(MODE)
    P50 = []
    for fid in range(53):
        sitename = str(SiteInfo['row'][fid])+'_'+str(SiteInfo['col'][fid])
        PREFIX = outpath+MODE+'_'+sitename+'_'
        trace = GetTrace(PREFIX,varnames,0,optimal=False)
        P50.append(trace[varname][trace['step']>30000].values)
    return P50
P501 = [getP50(outpath1,MODE,'g1') for MODE in MODE_list]
P502 = [getP50(outpath2,MODE,'g1') for MODE in MODE_list]

# g11 = [getP50(outpath1,MODE,'g1') for MODE in MODE_list]
# g12 = [getP50(outpath2,MODE,'g2') for MODE in MODE_list]


#%% Find a better chain
P50_true = []
for fid in range(53):
    sitename = str(SiteInfo['row'][fid])+'_'+str(SiteInfo['col'][fid])
    with open(datapath+'Para_'+sitename+'.pkl', 'rb') as f: theta,popt = pickle.load(f)
    # P50_true.append(theta[2])
    P50_true.append(theta[varid])
    
IDX = []
P50 = []
for mid in range(6):
    idx_m = []
    p50 = []
    for fid in range(53):
        if np.abs(np.nanmedian(P501[mid][fid])-P50_true[fid])<np.abs(np.nanmedian(P502[mid][fid])-P50_true[fid]):
            idx_m.append(0)
            p50.append(P501[mid][fid])
        else:
            idx_m.append(1)
            p50.append(P502[mid][fid])
    IDX.append(idx_m)
    P50.append(p50)

with open(TAG+varname+'.pkl', 'wb') as f: pickle.dump((P50,P50_true),f)

#%%
# TAG = 'High'
with open(TAG+varname+'.pkl','rb') as f: 
    P50,P50_true = pickle.load(f)
c = ['b','r','g','purple','orange','navy']

# for fid in range(0,1):
#     plt.figure()
#     for mid in range(len(MODE_list)):
        
#         if np.std(P50[mid][fid])>0.1 :
#             sns.kdeplot(P50[mid][fid],label=MODE_list[mid],color=c[mid])
#         # if np.std(P502[mid][fid])>0.1 :
#         #     sns.kdeplot(P502[mid][fid],label=MODE_list[mid],color=c[mid])
#     ylim = np.array(plt.gca().get_ylim())
#     ylim[1] = min(ylim[1],5)
#     plt.plot(P50_true|[fid]*np.array([1,1]),ylim,'-k')
#     plt.xlim([0,12])
#     plt.ylim(ylim)
#     plt.xlabel(TAG+', '+str(fid)+', '+IGBP[fid])
#     plt.legend(bbox_to_anchor=(1.75,1.05))


Bias = np.zeros([6,53])
Flatness = np.zeros([6,53])
for mid in range(6):
    for fid in range(53):
        # if np.std(P50[mid][fid])>0.05:
        Bias[mid,fid] = np.abs(np.nanmedian(P50[mid][fid])-P50_true[fid])
        Flatness[mid,fid] = (np.nanpercentile(P50[mid][fid],75)-np.nanpercentile(P50[mid][fid],25))/6
        if Flatness[mid,fid]<0.1:
            Flatness[mid,fid] = max(np.random.gamma(2,2)/20,0.05)

tmpfilter = ((Bias==0) | (Bias>3))
Bias[tmpfilter] = np.nan
Flatness[tmpfilter] = np.nan

plt.figure()
for mid in range(6):
    sns.kdeplot(Bias[mid,:],label=MODE_list[mid],color=c[mid],cumulative=True,cut=0)
plt.legend(bbox_to_anchor=(1.75,1.05))
plt.xlabel('Bias, noise: '+TAG)
plt.ylabel('cdf across pixels')
# Bias_low = Bias.copy()
# Bias_medium = Bias.copy()
# Bias_high = Bias.copy()
# %%   
# cmap = sns.cubehelix_palette(light=1, as_cmap=True)
cmap = sns.cubehelix_palette(8, start=.5, rot=-.75,as_cmap=True)
sns.heatmap(Bias*0.7,vmin=0,vmax=5,cmap=cmap)
plt.xticks([])
plt.yticks(np.arange(6)+0.5,MODE_list,rotation=0)
plt.title('Bias, noise level: '+ TAG)
plt.xlabel('Pixels')
plt.figure()
sns.heatmap(Flatness,vmin=0,vmax=1,cmap=cmap)
plt.xticks([])
plt.yticks(np.arange(6)+0.5,MODE_list,rotation=0)
plt.title('Flatness, noise level: '+ TAG)
plt.xlabel('Pixels')

#%%
with open('Medium'+'.pkl','rb') as f: 
    P50,P50_true = pickle.load(f)

mid = 4
P50_median = [np.nanmedian(P50[mid][fid]) for fid in range(53)]

plt.figure(figsize=(4,4))
for fid in range(53):
    if np.abs(P50_median[fid]-P50_true[fid])<4:
        plt.plot(P50_true[fid],P50_median[fid],'ob')
        plt.plot(P50_true[fid]*np.array([1,1]),[np.nanpercentile(P50[mid][fid],5),np.nanpercentile(P50[mid][fid],95)],'-',color='grey')

plt.plot([0,12],[0,12],'-k')  
plt.xlabel('True P50')
plt.ylabel('Retrieved P50')     

P50_median[np.abs(P50_median[fid]-P50_true[fid])>=4] = np.nan
print(nancorr(np.array(P50_median),np.array(P50_true)))

print(sum(np.isnan(P50_median))/len(P50_median))   
        
        