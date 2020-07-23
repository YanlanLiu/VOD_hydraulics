#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 14:48:03 2020

@author: yanlan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os; os.environ['PROJ_LIB'] = '/Users/yanlan/opt/anaconda3/pkgs/proj4-5.2.0-h0a44026_1/share/proj/'
from mpl_toolkits.basemap import Basemap
import sys; sys.path.append("../Utilities/")
from Utilities import LatLon
import seaborn as sns; sns.set(style="ticks", color_codes=True,font_scale=1.75)

TRY = pd.read_excel('../../TRY/Hydraulic_Traits_TRY_2.xls')
TRY_P50 = TRY['Water potential at 50% loss of conductivity Psi_50 (MPa)']
TRY_PFT = TRY['Plant funct type Decid angiosp=0 Evergr angiosp=1 Gymnosperm=2 Non-woody=3']
TRY = pd.DataFrame(np.column_stack([TRY_P50,TRY_PFT]),columns=['P50','PFT'])
TRY = TRY[TRY['P50']!=-999].reset_index()

TRY_P50_mean = [-np.mean(TRY['P50'][TRY['PFT']==itm]) for itm in [-1,0,2,3]]
TRY_P50_std = [np.std(TRY['P50'][TRY['PFT']==itm]) for itm in [-1,0,2,3]]
TRY_P50_min = [-np.nanpercentile(TRY['P50'][TRY['PFT']==itm],5) for itm in [-1,0,2,3]]
TRY_P50_max = [-np.nanpercentile(TRY['P50'][TRY['PFT']==itm],95) for itm in [-1,0,2,3]]

parentpath = '/Volumes/ELEMENTS/VOD_hydraulics/'
MODE = 'AM_PM_ET'
SiteInfo = pd.read_csv('SiteInfo_reps_50.csv')
Nsites = len(SiteInfo)
list(SiteInfo)
SiteInfo['IGBP']
IGBPlist = ['NA','ENF','EBF','DNF','DBF','DBF','Shrubland','Shrubland',
            'Savannas','Savannas','Grassland','Wetland','Cropland','Urban','Cropland','Snow']

IGBPnames = np.array([IGBPlist[itm] for itm in SiteInfo['IGBP'].values])
IGBPunique = np.unique(IGBPnames)

#%%
TAG = 'MC_ETwd'
statspath = parentpath + 'TroubleShooting/'+TAG+'/STATS/'
P50 = []; G1  = []; R2  = []

for fid in range(Nsites):
    sitename = str(SiteInfo['row'][fid])+'_'+str(SiteInfo['col'][fid])
    with open(statspath+MODE+'_'+sitename+'.pkl' , 'rb') as f: 
        TS_temporal_mean,TS_temporal_std,PARA_ensembel_mean,PARA_ensembel_std = pickle.load(f)  
    G1.append(PARA_ensembel_mean[1][0]+PARA_ensembel_std[1][0]*np.array([-1,0,1]))
    P50.append(PARA_ensembel_mean[1][2]+PARA_ensembel_std[1][2]*np.array([-1,0,1]))
    with open(statspath+'R2_'+sitename+'.pkl', 'rb') as f: 
        accen, r2_vod,r2_et,r2_sm,p50_pct, Geweke = pickle.load(f)
    R2.append([np.nanmax(r2_vod),np.nanmax(r2_et),np.nanmax(r2_sm)])
P50 = np.array(P50)
G1 = np.array(G1)
R2 = np.array(R2)

P50_PFT = [np.nanmedian(P50[IGBPnames==itm,:],axis=0) for itm in IGBPunique]
G1_PFT = [np.nanmedian(G1[IGBPnames==itm,:],axis=0) for itm in IGBPunique]


plt.figure(figsize=(8,8))
dd = 0.3
plt.subplot(211)
for i in range(len(IGBPunique)):
    if i==0:
        plt.bar(i-dd/2,P50_PFT[i][1],color='b',width=dd,label='Retrieved')
        plt.plot([i-dd/2,i-dd/2],[P50_PFT[i][0],P50_PFT[i][2]],'-k',label=r'$\pm$ std.')
        plt.bar(i+dd/2,TRY_P50_mean[i],color='r',width=dd,label=r'TRY')
    else:
        plt.bar(i-dd/2,P50_PFT[i][1],color='b',width=dd)
        plt.plot([i-dd/2,i-dd/2],[P50_PFT[i][0],P50_PFT[i][2]],'-k')
        plt.bar(i+dd/2,TRY_P50_mean[i],color='r',width=dd)
    plt.plot([i+dd/2,i+dd/2],TRY_P50_std[i]*np.array([-1,1])+TRY_P50_mean[i],'-k')
    
plt.xticks([])
plt.ylabel('-P50 (MPa)')
# plt.xlabel(TAG)
plt.legend(bbox_to_anchor=(1.05,1.05))


Lin_mean = np.array([5.79,3.97,2.35,4.5])

plt.subplot(212)
for i in range(len(IGBPunique)):
    if i==0:
        plt.bar(i-dd/2,G1_PFT[i][1],color='b',width=dd,label='Retrieved')
        plt.plot([i-dd/2,i-dd/2],[G1_PFT[i][0],G1_PFT[i][2]],'-k',label=r'$\pm$ std.')
        plt.bar(i+dd/2,Lin_mean[i],color='r',width=dd,label=r'Lin et al.')
    else:
        plt.bar(i-dd/2,G1_PFT[i][1],color='b',width=dd)
        plt.plot([i-dd/2,i-dd/2],[G1_PFT[i][0],G1_PFT[i][2]],'-k')
        plt.bar(i+dd/2,Lin_mean[i],color='r',width=dd)
    # plt.plot([i+dd/2,i+dd/2],TRY_P50_std[i]*np.array([-1,1])+TRY_P50_mean[i],'-k')

plt.xticks(np.arange(len(IGBPunique)),IGBPunique)
plt.ylabel('g1')
plt.xlabel(TAG)
plt.legend(bbox_to_anchor=(1.05,1.05))

#%%
lat,lon = LatLon(np.array(SiteInfo['row']),np.array(SiteInfo['col']))
mycmap = sns.cubehelix_palette(rot=-.63, as_cmap=True)

plt.figure(figsize=(13.2,20))
plt.subplot(411)
m = Basemap(llcrnrlon = -128, llcrnrlat = 25, urcrnrlon = -62, urcrnrlat = 50)
m.drawcoastlines(linewidth=1)
m.drawcountries(linewidth=1)
m.drawstates(linewidth=1)
# m.fillcontinents(zorder=0)
m.scatter(lon,lat,latlon=True,c=P50[:,1],s = (1-(P50[:,2]-P50[:,0])/7.5)*200,cmap = mycmap,vmin=0,vmax=13,zorder=10)
m.colorbar()
plt.title(TAG+', P50')

plt.subplot(412)
m = Basemap(llcrnrlon = -128, llcrnrlat = 25, urcrnrlon = -62, urcrnrlat = 50)
m.drawcoastlines(linewidth=1)
m.drawcountries(linewidth=1)
m.drawstates(linewidth=1)
m.scatter(lon,lat,latlon=True,c=R2[:,0],s=100,cmap = mycmap,vmin=0,vmax=0.7,zorder=10)
m.colorbar()
plt.title('R2_VOD')

plt.subplot(413)
m = Basemap(llcrnrlon = -128, llcrnrlat = 25, urcrnrlon = -62, urcrnrlat = 50)
m.drawcoastlines(linewidth=1)
m.drawcountries(linewidth=1)
m.drawstates(linewidth=1)
m.scatter(lon,lat,latlon=True,c=R2[:,1],s=100,cmap = mycmap,vmin=0.5,vmax=1,zorder=10)
m.colorbar()
plt.title('R2_ET')

plt.subplot(414)
m = Basemap(llcrnrlon = -128, llcrnrlat = 25, urcrnrlon = -62, urcrnrlat = 50)
m.drawcoastlines(linewidth=1)
m.drawcountries(linewidth=1)
m.drawstates(linewidth=1)
m.scatter(lon,lat,latlon=True,c=R2[:,2],s=100,cmap = mycmap,vmin=0,vmax=0.6,zorder=10)
m.colorbar()
plt.title('R2_SM')
#%%
# sns.boxplot(x='PFT',y='P50',data=TRY)
# plt.ylim([-15,.5])
# plt.xticks(np.arange(4),['DBF','EBF','ENF','Non-woody'])