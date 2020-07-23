#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 14:25:27 2020

@author: yanlan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 20:23:43 2020
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

parentpath = '/Volumes/ELEMENTS/VOD_hydraulics/'
versionpath = parentpath + 'Retrieval_0510/'
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
    parallels = np.arange(0.,81,2.)
    meridians = np.arange(10.,351.,5.)
    m.drawparallels(parallels,labels=[False,True,True,False])
    m.drawmeridians(meridians,labels=[True,False,False,True])

    # mycmap = sns.cubehelix_palette(6, rot=-.5, dark=.3,as_cmap=True)
    cs = m.pcolormesh(np.unique(lon),np.flipud(np.unique(lat)),heatmap1_data,cmap=mycmap,vmin=vmin,vmax=vmax,shading='quad')
    # cbar = m.colorbar(cs)
    # cbar.set_label(vname,rotation=360,labelpad=15)
    plt.show()
    

# V25,V50,V75,MissingList = ReadSpatialStats(versionpath+'Traits/Traits_')
# R2,RMSE,CORR, MissingListR2 = ReadSpatialStats(versionpath+'R2/R2_')
mLAI,mVOD,mET,N_VOD,N_ET = ReadSpatialStats(inpath+'Stats/Avg')

#%%
plotMap((V75[:,2]-V25[:,2])/7.5,varnames[2],vmin=0,vmax=1)
plotMap(R2[:,1],'R2_ET',vmin=0,vmax=1)
plotMap(mLAI,'LAI',vmin=0,vmax=3)
# plotMap(mVOD,'VOD',vmin=0,vmax=1)
# sum((R2[:,1]<0.1) & (mLAI<0.5))/sum((R2[:,1]<0.1))
# plt.xlabel('LAI');plt.ylabel('VOD')
#%%
tmp = pd.read_csv('../TroubleShooting/SiteInfo_reps_50.csv')
tmp = pd.concat([tmp,SiteInfo.iloc[[8568,8711,9143,9444,10844]]])
tmp = tmp.sort_values(by=['row','col']).reset_index().drop(columns=['level_0','Unnamed: 0','index','Unnamed: 0.1','Unnamed: 0.1.1'])
tmp.to_csv('SiteInfo_reps_55.csv')
#%% Select 100 representative pixels
# flatness = (V75[:,2]-V25[:,2])/7.5
# 
mLAI2 = mLAI*(mLAI>=0.7)
plotMap(mLAI2,'LAI',vmin=0,vmax=3)

lat,lon= LatLon(SiteInfo['row'],SiteInfo['col'])

idx = np.where((N_VOD>500) & (N_ET>50) & (lat>34) & (lat<38) & (lon>-115) & (lon<-105) & (mLAI>0.8) & (SiteInfo['IGBP'].values==1))[0]

lat[idx]
lon[idx]
SiteInfo['IGBP'].iloc[idx]
#%%
# plt.hist(flatness)
# R2[R2<-.2] = -.2
# plt.figure()
# plt.hist(R2[:,1],alpha=0.5)
# plt.hist(R2[:,0],alpha=0.5)


tmpdf = SiteInfo[['row','col']].sort_values(by='col')
tmpidx = np.array(tmpdf[(N_VOD>500) & (N_ET>50)].index)
repsidx = [tmpidx[i] for i in range(100,12100,120)]
repsidx.sort()

subset = SiteInfo.iloc[repsidx]
subset = subset.drop(columns=['Unnamed: 0.1'])

plt.figure();plt.hist(SiteInfo['row'],density=True,alpha=0.5);plt.hist(subset['row'],density=True,alpha=0.5)
plt.figure();plt.hist(SiteInfo['col'],density=True,alpha=0.5);plt.hist(subset['col'],density=True,alpha=0.5)
plt.figure();plt.hist(SiteInfo['IGBP'],density=True,alpha=0.5,bins=np.arange(16));plt.hist(subset['IGBP'],density=True,alpha=0.5,bins=np.arange(16))
plt.figure();plt.hist(SiteInfo['Soil texture'],density=True,alpha=0.5,bins=np.arange(16));plt.hist(subset['Soil texture'],density=True,alpha=0.5,bins=np.arange(16))
plt.figure();plt.hist(SiteInfo['Root type'],density=True,alpha=0.5,bins=np.arange(16));plt.hist(subset['Root type'],density=True,alpha=0.5,bins=np.arange(16))
plt.figure();plt.hist(SiteInfo['Root depth'],density=True,alpha=0.5,bins=np.arange(16));plt.hist(subset['Root depth'],density=True,alpha=0.5,bins=np.arange(16))

plt.figure();plt.hist(R2[:,1],density=True,alpha=0.5);plt.hist(R2[repsidx,1],density=True,alpha=0.5);plt.xlabel('R2_ET')
plt.figure();plt.hist(R2[:,0],density=True,alpha=0.5);plt.hist(R2[repsidx,0],density=True,alpha=0.5);plt.xlabel('R2_VOD')
plt.figure();plt.hist(flatness,density=True,alpha=0.5);plt.hist(flatness[repsidx],density=True,alpha=0.5);plt.xlabel('P50 flatness')
#%%
subset.to_csv('../Utilities/SiteInfo_reps.csv')