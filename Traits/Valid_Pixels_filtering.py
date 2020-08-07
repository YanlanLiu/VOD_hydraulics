#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 22:16:37 2020

@author: yanlan
"""

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
# yrange = np.arange(2003,2012)
# #datapath = 'D:/Data/VOD/AMSRE/'
# datapath = r"/Volumes/ELEMENTS/D/Data/VOD/AMSRE/"
varname = 'VOD_am_'
traitpath = r"/Volumes/ELEMENTS/VOD_hydraulics/TraitData/"
# #%%
# for year in yrange:
#     print(year)
#     vod_yr = np.load(datapath+'Annual/'+varname+str(year)+'.npy')  
#     vod_yr[vod_yr==-999] = np.nan
#     # vod_yr.shape
#     if year==yrange[0]:
#         VOD = np.nanmean(vod_yr,axis=2)
#     else:
#         VOD = VOD+np.nanmean(vod_yr,axis=2)
# VOD = VOD/len(yrange)
# np.save(varname+'avg.npy',VOD)

VOD = np.load(varname+'avg.npy')
plt.figure(figsize=(10,4))
plt.imshow(VOD,cmap='BrBG');plt.colorbar()
plt.title('Long-term average VOD_am')


#% Filter out pixels with too high and too low VOD
# plt.figure(figsize=(10,4))
# filter1 = (VOD>0.8); filter2 = (VOD<0.15)
# tmp = np.copy(VOD); tmp[filter1+filter2] = np.nan
# plt.imshow(tmp,cmap='BuGn');plt.colorbar()
# plt.xticks([]);plt.yticks([]);plt.clim([0,1])
# plt.title('Long-term avarage VOD, am')

# print(np.sum(~np.isnan(tmp)))
# print(np.sum(~np.isnan(VOD)))


#% Filter out pixels based on land cover
from netCDF4 import Dataset
f = Dataset(traitpath+'GLDASp4_domveg_025d.nc4')
domveg = np.flipud(f.variables['GLDAS_domveg'][0,:,:])#[160:260,200:470]  # US
plt.figure(); plt.imshow(domveg)
# filter_lc = ((domveg==0) | (domveg==11) | (domveg==13) | (domveg>14))
# domveg[filter_lc] = np.nan
# plt.figure();plt.imshow(domveg)
#%%
VOD_filtered = np.copy(VOD)
VOD_filtered[(VOD>0.8) | (VOD<0.15) | (domveg==0) | (domveg==11) | (domveg==13) | (domveg>14)] = np.nan
plt.figure(figsize=(10,4));plt.imshow(VOD_filtered,cmap='BuGn'); plt.colorbar()
plt.title('Filtered avg. VOD')
print('% of land with high VOD, low VOD, excluded LC:')
print(np.array([np.sum(VOD>0.8),np.sum(VOD<0.2),np.sum(1*((domveg==11) | (domveg==13) | (domveg>14)))])/(np.sum(domveg>0)))
print('% of remaining land area')
print(np.sum(~np.isnan(VOD_filtered))/np.sum(~np.isnan(VOD)))
print(f'Total # of pixels: {np.sum(~np.isnan(VOD_filtered)):1d}')
# row = np.array(SiteInfo['row']); col = np.array(SiteInfo['col'])
# IGBP = np.array([domveg[row[i],col[i]] for i in range(len(SiteInfo))])
# SiteInfo['IGBP'] = IGBP.astype('int')

#%%
r,c = np.where(~np.isnan(VOD_filtered))
SiteInfo_globe = pd.DataFrame({'row':r,'col':c})
SiteInfo_globe.to_csv('SiteInfo_globe.csv')

#%%
SiteInfo_US = pd.read_csv('SiteInfo_US.csv')

#%%
tmp = SiteInfo_globe.merge(SiteInfo_US, on=['row','col'], how='left', indicator=True).query('_merge == "left_only"').drop('_merge', 1)

