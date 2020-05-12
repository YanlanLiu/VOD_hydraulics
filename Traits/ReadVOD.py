#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 19:52:46 2020

@author: yanlan
"""

import numpy as np
from numpy import ma

import pandas as pd
import matplotlib.pyplot as plt

from datetime import date,datetime
from glob import glob
from netCDF4 import Dataset

arrayid = 0
yrange = np.arange(2003,2012)
#datapath = 'D:/Data/VOD/AMSRE/'
datapath = r"/Volumes/My Passport/D/Data/VOD/AMSRE/"
SiteInfo = np.array(pd.read_csv('SiteInfo_US.csv').iloc[arrayid*100:(arrayid+1)*100])[:,1:]
rlist = SiteInfo[:,0]; clist = SiteInfo[:,1]

def readVOD_ts(datapath,varname,rlist,clist):
    vod_ts = np.transpose(np.array([[] for i in range(len(rlist))]))
    for year in yrange:
        print(year)
        vod_yr = np.load(datapath+'Annual/'+varname+str(year)+'.npy')  
        tmp = np.transpose(np.array([vod_yr[rlist[i],clist[i],:] for i in range(len(rlist))]))
        vod_ts = np.concatenate([vod_ts,tmp],axis=0)
        vod_ts[vod_ts==-999] = np.nan
    return vod_ts

# soilm_am_ts = readVOD_ts(datapath,'SOILM_am_',rlist,clist)
# soilm_pm_ts = readVOD_ts(datapath,'SOILM_pm_',rlist,clist)
vod_am_ts = readVOD_ts(datapath,'VOD_am_',rlist,clist)
vod_pm_ts = readVOD_ts(datapath,'VOD_pm_',rlist,clist)
tt = np.arange(np.datetime64(str(yrange[0])+'-01-01'),np.datetime64(str(yrange[-1]+1)+'-01-01'))
#%%
for i in range(len(rlist)):
    tmp_df = pd.DataFrame({'VOD_am':vod_am_ts[:,i],'VOD_pm':vod_pm_ts[:,i]})
    tmp_df.to_csv('../AMSRE/VOD_'+str(rlist[i])+'_'+str(clist[i])+'.csv')