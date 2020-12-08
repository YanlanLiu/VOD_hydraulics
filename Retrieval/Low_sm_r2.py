#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 14:50:45 2020

@author: yanlan
"""


import os
import numpy as np
import pandas as pd
import glob
from newfun import GetTrace, get_var_bounds
import pickle
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="ticks", color_codes=True,font_scale=1.75)
import os; os.environ['PROJ_LIB'] = '/Users/yanlan/opt/anaconda3/pkgs/proj4-5.2.0-h0a44026_1/share/proj/'
from mpl_toolkits.basemap import Basemap
from Utilities import LatLon
parentpath = '/Volumes/ELEMENTS/VOD_hydraulics/'

nsites_per_id = 100
versionpath = parentpath + 'Global_0817/'
statspath = versionpath+'STATS/'
# statspath = versionpath+'STATS_bkp/'

MODE = 'VOD_SM_ET'
varnames, bounds = get_var_bounds(MODE)
SiteInfo = pd.read_csv('SiteInfo_globe_full.csv')
Collection_ACC = np.zeros([len(SiteInfo),4])+np.nan
Collection_OBS = np.zeros([len(SiteInfo),8])+np.nan

for arrayid in range(933):
    if np.mod(arrayid,100)==0:print(arrayid)
    subrange = np.arange(arrayid*nsites_per_id,min((arrayid+1)*nsites_per_id,len(SiteInfo)))
    fname = statspath+'EST_'+MODE+'_'+str(arrayid).zfill(3)+'.pkl'
    if os.path.isfile(fname):
        with open(fname,'rb') as f:
            TS_mean,TS_std,PARA_mean,PARA_std,ACC = pickle.load(f)
        if ACC.shape[1]>0:
            Collection_ACC[subrange,:] = ACC
            
    fname = statspath+'OBS_'+MODE+'_'+str(arrayid).zfill(3)+'.pkl'
    if os.path.isfile(fname):
        with open(fname,'rb') as f:
            OBSmean,OBSstd = pickle.load(f)
        if ACC.shape[1]>0:
            Collection_OBS[subrange,:] = np.copy(OBSmean)

#%%
mycmap = sns.cubehelix_palette(rot=-.63, as_cmap=True)
def plotmap(df,varname,vmin=0,vmax=1,cmap=mycmap):
    heatmap1_data = pd.pivot_table(df, values=varname, index='row', columns='col')
    heatmap1_data[578] = np.nan
    plt.figure(figsize=(13.2,5))
    m = Basemap(llcrnrlon = -180.0, llcrnrlat = -60.0, urcrnrlon = 180.0, urcrnrlat = 90.0)
    m.drawcoastlines()
    m.drawcountries()
    lat,lon = LatLon(np.array(heatmap1_data.index),np.array(list(heatmap1_data)))
    cs = m.pcolormesh(lon,lat,heatmap1_data,cmap=cmap,vmin=vmin,vmax=vmax,shading='flat')
    cbar = m.colorbar(cs)
    cbar.set_label(varname,rotation=360,labelpad=15)
    plt.show()
    return 0

#%%
df_acc = pd.DataFrame(Collection_ACC,columns=['r2_vod','r2_et','r2_sm','Geweke'])
df_acc['row'] = SiteInfo['row'];df_acc['col'] = SiteInfo['col']
plotmap(df_acc,'r2_vod')
plotmap(df_acc,'r2_et')
plotmap(df_acc,'r2_sm')
#%%
df_obs = pd.DataFrame(Collection_OBS,columns=['vod','et','soilm','rnet','temp','p','vpd','lai']) #VOD_ma,ET,SOILM,RNET,TEMP,P,VPD,LAI
df_obs['vpd'] = df_obs['vpd']*100
df_obs['temp'] = df_obs['temp']-273
df_obs['row'] = SiteInfo['row'];df_obs['col'] = SiteInfo['col']
plotmap(df_obs,'lai',vmin=0,vmax=3)
plotmap(df_obs,'temp',vmin=0,vmax=20)
plotmap(df_obs,'vpd',vmin=0,vmax=1e-2)

#%%
tsname = '/Volumes/ELEMENTS/VOD_hydraulics/Global_0817/Forward/TS_VOD_SM_ET_78_1257.pkl'
with open(tsname,'rb') as f:
    VODhat,EThat,PSILhat,S1hat = pickle.load(f)
#%%
from Utilities import IsOutlier

mS1hat = np.nanmean(S1hat,axis=0)
validsm = ~(np.isnan(SOILM)+IsOutlier(SOILM,multiplier=2))

plt.plot(mS1hat[validsm])
plt.plot(SOILM[validsm])
print(np.corrcoef(SOILM[validsm],mS1hat[validsm])[0,1]**2)
# TS = [np.reshape(itm,[nsample,-1]) for itm in TS]
