#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 14:35:59 2020

@author: yanlanliu
"""

import os
import numpy as np
import pandas as pd
import glob
from newfun import GetTrace, get_var_bounds
import pickle
from matplotlib import colors
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="ticks", color_codes=True,font_scale=1.75)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from Utilities import LatLon
parentpath = '/Volumes/ELEMENTS/VOD_hydraulics/'

nsites_per_id = 100
versionpath = parentpath + 'Global_0817/'
#statspath = versionpath+'STATS/'
statspath = versionpath+'STATS_bkp/STATS/'

MODE = 'VOD_SM_ET'
varnames, bounds = get_var_bounds(MODE)
SiteInfo = pd.read_csv('SiteInfo_globe_full.csv')
Collection_ACC = np.zeros([len(SiteInfo),4])+np.nan
Collection_PARA = np.zeros([len(SiteInfo),17])+np.nan
Collection_STD = np.zeros([len(SiteInfo),11])+np.nan

Collection_OBS = np.zeros([len(SiteInfo),9])+np.nan
Collection_N = np.zeros([len(SiteInfo),3])+np.nan

for arrayid in range(933):
    if np.mod(arrayid,100)==0:print(arrayid)
    subrange = np.arange(arrayid*nsites_per_id,min((arrayid+1)*nsites_per_id,len(SiteInfo)))
    fname = statspath+'EST_'+MODE+'_'+str(arrayid).zfill(3)+'.pkl'
    if os.path.isfile(fname):
        with open(fname,'rb') as f:
            TS_mean,TS_std,PARA_mean,PARA_std,PARA2_mean,PARA2_std,ACC = pickle.load(f)
        if ACC.shape[1]>0:
            Collection_ACC[subrange,:] = ACC
            Collection_PARA[subrange,:] = PARA_mean
            Collection_STD[subrange,:] = PARA2_std
    fname = statspath+'OBS_'+MODE+'_'+str(arrayid).zfill(3)+'.pkl'
    if os.path.isfile(fname):
        with open(fname,'rb') as f:
            OBS_mean,OBS_std,OBS_N = pickle.load(f)            
        if OBS_mean.shape[1]>0:
            Collection_OBS[subrange,:] = OBS_mean
            Collection_N[subrange,:] = OBS_N            
            


df_para = pd.DataFrame(Collection_PARA[:,3:],columns=varnames+['a','b','c'])
df_para['row'] = SiteInfo['row'];df_para['col'] = SiteInfo['col']; df_para['IGBP'] = SiteInfo['IGBP']
df_para['psi50X'] = -df_para['psi50X']
df_para['lpx'] = df_para['lpx']*df_para['psi50X']

#%%
import netCDF4
from scipy import ndimage
IGBPlist = ['NA','ENF','EBF','DNF','DBF','MF','SHB','SHB',
            'SAV','SAV','GRA','WET','CRO','URB','GRA','SNW','NA','NA','NA']

IGBPnames = np.array([IGBPlist[itm] for itm in df_para['IGBP'].values])

fp='../CONUS/Trugman_map/CWM_P50_10Deg.nc'
nc = netCDF4.Dataset(fp)
lat = np.array(nc['lat'][:])
lon = np.array(nc['lon'][:])
p50_att = np.array(nc['CWM_P50'][:])
nplots  = np.array(nc['nplots'][:])
lat_2d = np.tile(lat,[len(lon),1])
lon_2d = np.transpose(np.tile(lon,[len(lat),1]))
fia = pd.DataFrame({'Lat':np.reshape(lat_2d,[-1,]),'Lon':np.reshape(lon_2d,[-1,]),'P50':np.reshape(p50_att,[-1,]),'nplots':np.reshape(nplots,[-1,])})
fia = fia.dropna().reset_index()

lat0,lon0 = LatLon(df_para['row'].values,df_para['col'].values)
psi50x = df_para['psi50X'].values

EST = []; igbp = []
for i in range(len(fia)):
    tmp = fia.iloc[i]
    idx = np.where((lat0-tmp['Lat'])**2+(lon0-tmp['Lon'])**2<= 2*(0.5-0.125)**2)[0]
    if len(idx)>0:
        subp50 = psi50x[idx]
        subigbp = list(IGBPnames[idx])
        N = sum(~np.isnan(subp50)); mp50 = np.nanmean(subp50); stdp50 = np.nanstd(subp50)
        igbp.append(max(subigbp, key=subigbp.count))
        EST.append([mp50,N,stdp50])
    else:
        EST.append([np.nan,np.nan,np.nan])
        igbp.append(np.nan)
EST = np.array(EST)
fia['P50_hat'] = EST[:,0]; fia['Nhat'] = EST[:,1]; fia['P50_std'] = EST[:,2];fia['IGBPnames'] = igbp   
igbpfilter = [itm not in ['CRO','GRA'] for itm in fia['IGBPnames']]
fia = fia[(fia['Nhat']>4) & igbpfilter]

#%%

np.random.seed(0) 
# fig = plt.figure(figsize=(6,6))
# ax = plt.subplot(111)

# xlim = [-9.9,-0.4]
xlim = [-9.4,-0.4]

fia_s = fia[fia['IGBPnames']=='ENF']
for igbp in ['EBF','DBF','MF']:
    fia_s = pd.concat([fia_s,fia[fia['IGBPnames']==igbp]],axis=0)

fia_s['Land cover'] = fia_s['IGBPnames']

# fia_s.drop(fia_s[(fia_s['P50']<-7)&(fia_s['P50_hat']>-4)].index,inplace=True)
fia_s = fia_s.reset_index()

fia_s0 = fia_s.copy()
fia_s0['rnd'] = np.random.randint(len(fia_s0),size=len(fia_s0))
fia_s0 = fia_s0.sort_values(by='rnd')
fia_s0['Land cover'][fia_s0['Land cover']=='MF'] = 'Mixed forest'
fia_s0['Land cover'][fia_s0['Land cover']=='ENF'] = 'Evergreen needleleaf forest'
fia_s0['Land cover'][fia_s0['Land cover']=='DBF'] = 'Deciduous broadleaf forest'
fia_s0['Land cover'][fia_s0['Land cover']=='EBF'] = 'Evergreen broadleaf forest'


# sc = sns.scatterplot(x="P50", y="P50_hat",s=nplots*0.8,alpha=0.5, hue="Land cover",data=fia_s,palette='colorblind',ax=ax)
plt.figure(figsize=(6,6))
plt.plot(xlim,xlim,'--k')

sc = sns.scatterplot(x="P50", y="P50_hat",s=fia_s0['nplots']*0.5+10,alpha=0.7, hue="Land cover",
                     hue_order = ['Evergreen needleleaf forest','Evergreen broadleaf forest','Deciduous broadleaf forest','Mixed forest'],
                     data=fia_s0,palette='cubehelix')
plt.legend(bbox_to_anchor=(1.05,1.03),title='')

# handles, labels = ax.get_legend_handles_labels()
# ax.legend_.remove()
# l = sc.legend(handles[1:],labels[1:], ncol=1,loc=3)
# # ax.legend(handles=handles[1:], labels=labels[1:])

# frame = l.get_frame()
# frame.set_alpha(1)
# frame.set_edgecolor('k')

# plt.plot([-7,-0.4],[-7,-0.4],'--k');plt.xlim(xlim);plt.ylim(xlim)
plt.xlim(xlim);plt.ylim(xlim)
plt.xlabel(r'FIA-based $\psi_{50,x}$ (MPa)')
plt.ylabel(r'Estimated $\psi_{50,x}$ (MPa)')
plt.savefig('../Figures/Fig6_p50_fia.png',dpi=500,bbox_inches='tight')
