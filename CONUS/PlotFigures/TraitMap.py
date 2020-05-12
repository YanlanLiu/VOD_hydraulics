#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 19:21:28 2020

@author: yanlan
"""

import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
import seaborn as sns; sns.set(style="ticks", color_codes=True,font_scale=1.75)

parentpath = '/Volumes/ELEMENTS/VOD_hydraulics/'
versionpath = parentpath + 'Retrieval_0501/'
outpath = versionpath+'Output/'

traitpath = versionpath+'Traits/'
r2path = versionpath+'R2_test/'
npath = parentpath+'Input/ValidN/'
SiteInfo = pd.read_csv('SiteInfo_US_full.csv')

varlist = ['g1','lpx','psi50X','gpmax','C','bexp','bc']

MODE = 'AM_PM_ET_'

#%%
V50 = np.zeros([0,len(varlist)]); 
V25 = np.copy(V50); V75 = np.copy(V50)
ML = np.zeros([0,])
    
R2 = np.zeros([0,2])
ValidN = np.zeros([len(SiteInfo),2])
for arrayid in range(14):
    print(arrayid)
    traitname = traitpath+'Traits_'+str(arrayid)+'_1E3.pkl'

    with open(traitname, 'rb') as f: 
        Val_25, Val_50, Val_75, MissingList = pickle.load(f)
    V25 = np.concatenate([V25,Val_25],axis=0)
    V50 = np.concatenate([V50,Val_50],axis=0)
    V75 = np.concatenate([V75,Val_75],axis=0)
    ML = np.concatenate([ML,MissingList])
    
    nname = npath+'N_'+str(arrayid)+'_1E3.pkl.npy'
    vn = np.load(nname)
    array_range = np.arange(arrayid*1000,(arrayid+1)*1000)
    ValidN[array_range,:] =  vn[array_range,:]
    # plt.figure();plt.plot(vn[:,1])
    r2anme = r2path+'R2_'+str(arrayid)+'_1E3.pkl'
    with open(r2anme, 'rb') as f: 
        r2, MissingListR2 = pickle.load(f)
    R2 = np.concatenate([R2,r2],axis=0)
    
#%%
# SiteInfo = SiteInfo.iloc[0:len(V25)]
# # Trait = pd.DataFrame((V75-V25)/V50,columns=varlist)

Trait = pd.DataFrame(V50,columns=varlist)
Acc = pd.DataFrame(R2,columns=['R2_VOD','R2_ET'])
VN = pd.DataFrame(ValidN,columns=['N_VOD','N_ET'])

df = pd.concat([SiteInfo,Trait,Acc,VN],axis=1)

#%%
varname = 'Soil texture'
heatmap1_data = pd.pivot_table(df, values=varname, index='row', columns='col')
plt.figure(figsize=(13.5,5))
# plt.imshow(heatmap1_data,cmap='Greens'); plt.colorbar();plt.xticks([]);plt.yticks([])
plt.imshow(heatmap1_data,cmap='summer_r');
plt.clim([0,10]);plt.colorbar()
plt.title(varname)

#%%
BB = np.array([4.05,4.38,4.90,5.3,5.39,7.12,7.75,8.52,10.4,10.4,11.4])
CH_bexp = [BB[int(tx)] for tx in np.array(SiteInfo['Soil texture'])]
def Saxton(pct): # pct -- [%sand, %silt, %clay]; th -- volumetric soil moisture
    S,C = (pct[0],pct[2])
    e = -3.140
    f = -2.22e-3
    g = -3.484e-5
    B = e+f*C**2+g*S**2+g*S**2*C
    return B

SS = list(np.array(df[['T_SAND','T_SILT','T_CLAY']]))
S_bexp = [-Saxton(pct) for pct in SS]

#%%
plt.plot(df['bexp'],S_bexp,'ok')
plt.ylabel('HWSD, Saxton et al.')
plt.xlabel('Retrieved')

#%%
# df['R2_ET'][df['R2_ET']<0] = 0
subset = df[df['IGBP'].isin([1,4,5,7,9,10])*(df['R2_VOD']>-1)]

IGBPlist = ['NA','ENF','EBF','DNF','DBF','DBF','Shrublands','Shrublands',
            'Savannas','Savannas','Grassland','Wetland','Cropland']
IGBP_str = [IGBPlist[itm] for itm in np.array(subset['IGBP'])]
subset['IGBP_str'] = IGBP_str

plt.figure(figsize=(10,4))
sns.violinplot(x='IGBP_str',y='R2_VOD',data=subset.sort_values(by=['IGBP']))
plt.ylim([-0.25,0.8])

plt.xlabel('')
print(df['R2_VOD'].quantile(.25),df['R2_VOD'].quantile(.75))
print(df['R2_ET'][df['R2_ET']>-1].quantile(.25),df['R2_ET'][df['R2_ET']>-1].quantile(.75))

#%%
from scipy.stats import gaussian_kde
x = np.array(df['C']); y = np.array(SiteInfo['LAI_50'])
tmpfilter = ~np.isnan(x+y); x= x[tmpfilter]; y = y[tmpfilter]
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)

plt.scatter(x,y,c=z)
plt.xlabel('C')
plt.ylabel('Annual median LAI')

#%% Identify negative R2_ET

idx = np.where((df['R2_ET']<0)*(df['R2_VOD']>0))[0]

sitename = [str(SiteInfo['row'][i])+'_'+str(SiteInfo['col'][i]) for i in idx]

sitename = sitename[1000]
#%%
from newfun import readCLM,GetTrace
import os
inpath = parentpath+'Input/'
forwardpath = versionpath+'Forward/'
Forcings,VOD,ET,dLAI,discard_vod,discard_et,idx = readCLM(inpath,sitename)
forwardname = forwardpath+MODE+sitename+'.pkl'
if os.path.isfile(forwardname):
    with open(forwardname,'rb') as f:  # Python 3: open(..., 'rb')
        SVOD, SET, SPSIL = pickle.load(f)

PREFIX = outpath+MODE+sitename+'_'
trace = GetTrace(PREFIX,0,optimal=False)

#%%
plt.plot(ET,'or')
plt.plot(np.min(SET,axis=0),color='lightblue')
plt.plot(np.max(SET,axis=0),color='lightblue')
plt.plot(np.mean(SET,axis=0),color='navy')
plt.ylabel('ET')

plt.figure()
plt.plot(VOD,'or')
plt.plot(np.min(SVOD,axis=0),color='lightblue')
plt.plot(np.max(SVOD,axis=0),color='lightblue')
plt.plot(np.mean(SVOD,axis=0),color='navy')
plt.ylabel('VOD')

#%%
plt.figure()
plt.plot(trace['sigma_vod'])