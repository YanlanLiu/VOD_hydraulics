#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 15:27:49 2020

@author: yanlan

Plot average VOD pattern
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
yrange = np.arange(2003,2012)
#datapath = 'D:/Data/VOD/AMSRE/'
datapath = r"/Volumes/ELEMENTS/D/Data/VOD/AMSRE/"
varname = 'VOD_am_'
#%%
for year in yrange:
    print(year)
    vod_yr = np.load(datapath+'Annual/'+varname+str(year)+'.npy')  
    vod_yr[vod_yr==-999] = np.nan
    # vod_yr.shape
    if year==yrange[0]:
        VOD = np.nanmean(vod_yr,axis=2)
    else:
        VOD = VOD+np.nanmean(vod_yr,axis=2)
VOD = VOD/len(yrange)
np.save(varname+'avg.npy',VOD)
#%%
VOD = np.load(varname+'avg.npy')
plt.figure(figsize=(10,4))
plt.imshow(VOD,cmap='BrBG');plt.colorbar()
# plt.xticks([]);plt.yticks([]);plt.clim([0,1.1])
plt.title('Long-term average VOD_am')
#%%
incidence_angle = 55
transmissivity = np.exp(-VOD/np.cos(incidence_angle/180*np.pi))
plt.figure(figsize=(10,4))
plt.imshow(transmissivity,cmap='BrBG_r');plt.colorbar()
plt.xticks([]);plt.yticks([]);plt.clim([0,1])
plt.title('Transmissivity')

#%%
plt.figure(figsize=(10,4))
filter1 = (VOD>0.8); filter2 = (VOD<0.15)
tmp = np.copy(VOD); tmp[filter1+filter2] = np.nan
plt.imshow(tmp,cmap='BuGn');plt.colorbar()
plt.xticks([]);plt.yticks([]);plt.clim([0,1])
plt.title('Long-term avarage VOD, am')

print(np.sum(~np.isnan(tmp)))
print(np.sum(~np.isnan(VOD)))
#%%
VOD_US = VOD[160:260,200:470]
plt.imshow(VOD_US)
print(np.sum(~np.isnan(VOD_US)))
print(np.sum(~np.isnan(tmp[160:260,200:470])))

tmp0 = np.copy(VOD)+np.nan
tmp0[160:260,200:470] = np.copy(tmp[160:260,200:470])
r,c = np.where(~np.isnan(tmp0))
SiteInfo_US = pd.DataFrame({'row':r,'col':c})
SiteInfo_US.to_csv('SiteInfo_US.csv')
#%%
plt.figure()
tmp  = np.reshape(transmissivity,[-1,])
plt.hist(tmp[~np.isnan(tmp)],color='grey')
plt.plot([0.25,0.25],[0,50000],'--r')
plt.xlabel('Transmissivity')
plt.ylabel('Frequency')

plt.figure()
tmp  = np.reshape(VOD,[-1,])
plt.hist(tmp[~np.isnan(tmp)],color='grey')
plt.plot([0.15,0.15],[0,40000],'--r')
plt.xlabel('VOD')
plt.ylabel('Frequency')