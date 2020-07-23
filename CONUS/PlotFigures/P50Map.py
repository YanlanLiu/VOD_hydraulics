#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 17:43:02 2020

@author: yanlan
"""

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
import os; os.environ['PROJ_LIB'] = '/Users/yanlan/opt/anaconda3/pkgs/proj4-5.2.0-h0a44026_1/share/proj/'
from mpl_toolkits.basemap import Basemap
import sys; sys.path.append("../Utilities/")
from Utilities import LatLon
from scipy.stats import norm,gamma
from newfun import varnames

parentpath = '/Volumes/ELEMENTS/VOD_hydraulics/'
versionpath = parentpath + 'Retrieval_0705/'
# versionpath = parentpath + 'SM_0717/'

SiteInfo = pd.read_csv('../Utilities/SiteInfo_US_full.csv').iloc[:14000]
inpath = parentpath+'Input/'
LOC = SiteInfo[['row','col']]

def ReadSpatialStats(PREFIX,SUFFIX='_1E3.pkl',N=14):
    fname = PREFIX+str(0)+SUFFIX
    with open(fname, 'rb') as f: 
        VAL = pickle.load(f)
    for arrayid in range(1,N):
        fname = PREFIX+str(arrayid)+SUFFIX
        with open(fname, 'rb') as f: 
            val = pickle.load(f)
        if len(val)<10:
            for ii in range(len(val)):
                VAL[ii] = np.concatenate([VAL[ii],val[ii]],axis=0)
        else:
            VAL = np.concatenate([VAL,val],axis=0)
    return VAL

def plotMap(varray,vname,vmin=0,vmax=1):
    LOC[vname] = varray
    df_loc = LOC.dropna()
    lat,lon = LatLon(np.array(df_loc['row']),np.array(df_loc['col']))
    heatmap1_data = pd.pivot_table(df_loc,values=vname,index='row', columns='col')
    plt.figure(figsize=(13.2,5))
    m = Basemap(llcrnrlon = -128, llcrnrlat = 25, urcrnrlon = -62, urcrnrlat = 50)
    m.drawcoastlines()
    m.drawcountries()
    mycmap = sns.cubehelix_palette(rot=-.63, as_cmap=True)
    # mycmap = sns.cubehelix_palette(6, rot=-.5, dark=.3,as_cmap=True)
    cs = m.pcolormesh(np.unique(lon),np.flipud(np.unique(lat)),heatmap1_data,cmap=mycmap,vmin=vmin,vmax=vmax,shading='quad')
    cbar = m.colorbar(cs)
    cbar.set_label(vname,rotation=360,labelpad=15)
    plt.show()
    

V25,V50,V75,Geweke,MissingList = ReadSpatialStats(versionpath+'Traits/Traits_')
# R2,RMSE,CORR, MissingListR2 = ReadSpatialStats(versionpath+'R2/R2_')
mLAI,mVOD,mET,N_VOD,N_ET = ReadSpatialStats(inpath+'Stats/Avg')

VN = pd.DataFrame(np.column_stack([N_VOD,N_ET]),columns=['N_VOD','N_ET'])

# df['obsfilter'] = (df['N_VOD']>10) & (df['N_ET']>2)*1 # to be used
# df = df[df['obsfilter']==1]

#%%
Trait = pd.DataFrame(V50,columns=varnames[:V50.shape[1]])

df = pd.concat([SiteInfo,Trait,VN],axis=1)
df['obsfilter'] = (df['N_VOD']>10) & (df['N_ET']>2)*1 # to be used
df['Geweke'] = Geweke
df = df[df['obsfilter']==1]

df = df[(df['obsfilter']==1) & (df['Geweke']<0.3)]

# varname = 'g1'
varname='psi50X'
# if df[varname].mean()>0:df[varname] = -df[varname]
lat,lon = LatLon(np.array(df['row']),np.array(df['col']))

heatmap1_data = pd.pivot_table(df, values=varname, index='row', columns='col')
fig=plt.figure(figsize=(13.2,5))
m = Basemap(llcrnrlon = -128, llcrnrlat = 25, urcrnrlon = -62, urcrnrlat = 50)

m.drawcoastlines(linewidth=1)
m.drawcountries(linewidth=1)
m.drawstates(linewidth=1)
mycmap = sns.cubehelix_palette(rot=-.63, as_cmap=True)
# mycmap = sns.cubehelix_palette(6, rot=-.5, dark=.3,as_cmap=True)
cs = m.pcolormesh(np.unique(lon),np.flipud(np.unique(lat)),heatmap1_data,cmap=mycmap,vmin=0,vmax=15,shading='quad')
cbar = m.colorbar(cs)
cbar.set_label(varname,rotation=360,labelpad=15)
plt.show()

#%%
IGBPlist = ['NA','ENF','EBF','DNF','DBF','DBF','Shrubland','Shrubland',
            'Savannas','Savannas','Grassland','Wetland','Cropland','Urban','Cropland','Snow','NA','NA','NA']

IGBPnames = np.array([IGBPlist[itm] for itm in df['IGBP'].values])
IGBPunique = np.unique(IGBPnames)

P50mean = [np.nanmean(df['psi50X'][IGBPnames==itm]) for itm in ['Cropland','DBF','ENF','Grassland']]
P50std = [np.nanstd(df['psi50X'][IGBPnames==itm]) for itm in ['Cropland','DBF','ENF','Grassland']]

plt.figure(figsize=(8,8))
dd = 0.3
plt.subplot(211)
for i in range(4):
    if i==0:
        plt.bar(i-dd/2,P50mean[i],color='b',width=dd,label='Retrieved')
        plt.plot([i-dd/2,i-dd/2],P50mean[i]+P50std[i]*np.array([-1,1]),'-k',label=r'$\pm$ std.')
        plt.bar(i+dd/2,TRY_P50_mean[i],color='r',width=dd,label=r'TRY')
    else:
        plt.bar(i-dd/2,P50mean[i],color='b',width=dd)
        plt.plot([i-dd/2,i-dd/2],P50mean[i]+P50std[i]*np.array([-1,1]),'-k')
        plt.bar(i+dd/2,TRY_P50_mean[i],color='r',width=dd)
    plt.plot([i+dd/2,i+dd/2],TRY_P50_std[i]*np.array([-1,1])+TRY_P50_mean[i],'-k')
    
plt.xticks([])
plt.ylabel('-P50 (MPa)')
plt.xticks(np.arange(4),['Cropland','DBF','ENF','Grassland'])

# plt.xlabel(TAG)
plt.legend(bbox_to_anchor=(1.05,1.05))


Lin_mean = np.array([2.35,3.97,4.22,4.5,5.76,5.79])
G1mean = [np.nanmean(df['g1'][IGBPnames==itm]) for itm in ['ENF','DBF','Shrubland','Grassland','Savannas','Cropland']]
G1std = [np.nanstd(df['g1'][IGBPnames==itm]) for itm in ['ENF','DBF','Shrubland','Grassland','Savannas','Cropland']]

plt.subplot(212)
for i in range(6):
    if i==0:
        plt.bar(i-dd/2,G1mean[i],color='b',width=dd,label='Retrieved')
        plt.plot([i-dd/2,i-dd/2],G1mean[i]+G1std[i]*np.array([-1,1]),'-k',label=r'$\pm$ std.')
        plt.bar(i+dd/2,Lin_mean[i],color='r',width=dd,label=r'Lin et al.')
    else:
        plt.bar(i-dd/2,G1mean[i],color='b',width=dd)
        plt.plot([i-dd/2,i-dd/2],G1mean[i]+G1std[i]*np.array([-1,1]),'-k')
        plt.bar(i+dd/2,Lin_mean[i],color='r',width=dd)
    # plt.plot([i+dd/2,i+dd/2],TRY_P50_std[i]*np.array([-1,1])+TRY_P50_mean[i],'-k')

plt.xticks(np.arange(6),['ENF','DBF','Shrubland','Grassland','Savannas','Cropland'],rotation=30)
plt.ylabel('g1')
plt.xlabel('MC_SM2')
plt.legend(bbox_to_anchor=(1.05,1.05))

#%%
df_trees = df[(IGBPnames!='Grassland')]

# varname = 'g1'
varname='psi50X'
# if df[varname].mean()>0:df[varname] = -df[varname]
lat,lon = LatLon(np.array(df_trees['row']),np.array(df_trees['col']))

heatmap1_data = pd.pivot_table(df_trees, values=varname, index='row', columns='col')
fig=plt.figure(figsize=(13.2,5))
m = Basemap(llcrnrlon = -128, llcrnrlat = 25, urcrnrlon = -62, urcrnrlat = 50)

m.drawcoastlines(linewidth=1)
m.drawcountries(linewidth=1)
m.drawstates(linewidth=1)
mycmap = sns.cubehelix_palette(rot=-.63, as_cmap=True)
# mycmap = sns.cubehelix_palette(6, rot=-.5, dark=.3,as_cmap=True)
cs = m.pcolormesh(np.unique(lon),np.flipud(np.unique(lat)),heatmap1_data,cmap=mycmap,vmin=0,vmax=12,shading='quad')
cbar = m.colorbar(cs)
cbar.set_label(varname,rotation=360,labelpad=15)
plt.show()