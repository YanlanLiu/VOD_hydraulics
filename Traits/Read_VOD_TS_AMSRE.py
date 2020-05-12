# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 20:58:19 2020

@author: yanlan

Read VOD data and save as time series
"""
import numpy as np
from numpy import ma

import pandas as pd
import matplotlib.pyplot as plt

from datetime import date,datetime
from glob import glob
from netCDF4 import Dataset

yrange = np.arange(2003,2012)
#datapath = 'D:/Data/VOD/AMSRE/'
datapath = r"/Volumes/My Passport/D/Data/VOD/AMSRE/"
#%%
ampm = 'pm'
t = date(2006,6,19)
def readVOD_image(datapath,t,ampm,masked=True):
    if ampm=='am':PREFIX = 'LPRM-AMSR_E_L3_D_SOILM3_V002_' # 1:30 am
    elif ampm=='pm':PREFIX = 'LPRM-AMSR_E_L3_A_SOILM3_V002_' # 1:30 pm
    else: print('Invalid ampm')
    fname = glob(datapath+PREFIX+t.strftime('%Y%m%d')+'*.nc')
    if len(fname)>0:
        f = Dataset(fname[0])
        soilm = np.transpose(f.variables['soil_moisture_x'][:,:600])
        vod = np.transpose(f.variables['opt_depth_x'][:,:600])
    else:
        soilm = ma.masked_array(np.zeros([600,1440]),mask = np.ones([600,1440]))
        vod = soilm.copy()
    if masked==False:
        soilm = ma.filled(soilm,-999)
        vod = ma.filled(vod,-999)
    return soilm, vod

soilm,vod = readVOD_image(datapath,t,ampm,masked=False)
#plt.imshow(vod)
#vod[vod==-999] = np.nan;plt.imshow(vod)
##%%
for year in yrange:
    print(year)
    trange = np.arange(np.datetime64(str(year)+'-01-01'),np.datetime64(str(year+1)+'-01-01'))
    vod_am = np.zeros([vod.shape[0],vod.shape[1],len(trange)])+np.nan
    vod_pm = vod_am.copy()
    soilm_am = vod_am.copy()
    soilm_pm = vod_am.copy()
    for t in trange:
        yday = t.astype(datetime).timetuple().tm_yday
        soilm_am[:,:,yday-1],vod_am[:,:,yday-1] = readVOD_image(datapath,t.astype(datetime),'am',masked=False)
        soilm_pm[:,:,yday-1],vod_pm[:,:,yday-1] = readVOD_image(datapath,t.astype(datetime),'pm',masked=False)
    np.save(datapath+'Annual/VOD_am_'+str(year)+'.npy',vod_am)
    np.save(datapath+'Annual/VOD_pm_'+str(year)+'.npy',vod_pm)
    np.save(datapath+'Annual/SOILM_am_'+str(year)+'.npy',soilm_am)
    np.save(datapath+'Annual/SOILM_pm_'+str(year)+'.npy',soilm_pm)
    
#%% Start from here for other pixels
#% Read VOD time series from images
#rlist = [290,433]; clist = [1045,861]
rlist = [509]; clist = [1296]
def readVOD_ts(datapath,varname,rlist,clist):
    vod_ts = np.transpose(np.array([[] for i in range(len(rlist))]))
    for year in yrange:
        print(year)
        vod_yr = np.load(datapath+'Annual/'+varname+str(year)+'.npy')  
        tmp = np.transpose(np.array([vod_yr[rlist[i],clist[i],:] for i in range(len(rlist))]))
        vod_ts = np.concatenate([vod_ts,tmp],axis=0)
        vod_ts[vod_ts==-999] = np.nan
    return vod_ts

soilm_am_ts = readVOD_ts(datapath,'SOILM_am_',rlist,clist)
soilm_pm_ts = readVOD_ts(datapath,'SOILM_pm_',rlist,clist)
vod_am_ts = readVOD_ts(datapath,'VOD_am_',rlist,clist)
vod_pm_ts = readVOD_ts(datapath,'VOD_pm_',rlist,clist)
tt = np.arange(np.datetime64(str(yrange[0])+'-01-01'),np.datetime64(str(yrange[-1]+1)+'-01-01'))
#%%
for i in range(len(rlist)):
    tmp_df = pd.DataFrame({'Time':tt,'SOILM_am':soilm_am_ts[:,i],'SOILM_pm':soilm_pm_ts[:,i],'VOD_am':vod_am_ts[:,i],'VOD_pm':vod_pm_ts[:,i]})
    tmp_df.to_csv('../AMSRE/SOILM_VOD_'+str(rlist[i])+'_'+str(clist[i])+'.csv')