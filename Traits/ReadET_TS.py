# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 16:03:36 2020

@author: yanlan

Read ET
"""

import numpy as np
from numpy import ma

import pandas as pd
import matplotlib.pyplot as plt

from datetime import date,datetime
import glob
import os
#for year in yrange:
#    print(year)
#    trange = np.arange(np.datetime64(str(year)+'-01-01'),np.datetime64(str(year+1)+'-01-01'))

#yrange = np.arange(2011,2014)
yrange = np.arange(2003,2014)
#%%
#inpath = 'D:/Data/ET/ALEXIET/'
#flist = glob.glob(inpath+'*_2003*.npy')
#week = 0
#tmp = np.load(PREFIX+str(y)+str(8+week*7).zfill(3)+'.npy')

PREFIX = 'D:/Data/ET/ALEXIET/EDAY_CFSR_'
for y in yrange:
    print(y)
    anET = np.zeros([600,1440,52])+np.nan
    for week in range(0,52):
        fname = PREFIX+str(y)+str(8+week*7).zfill(3)+'.npy'
        if os.path.isfile(fname):
            anET[:,:,week] = np.load(fname)
    np.save('D:/Data/ET/ALEXIET_annual/ET'+str(y)+'.npy',anET)
    

#plt.figure(figsize=(10,4))
#plt.imshow(np.mean(anET,axis=2)*365);plt.colorbar();plt.clim([0,2000])
#%% start from here for other pixels
#rlist = [290,433]; clist = [1045,861]
rlist = [509]; clist = [1296]

yrange = np.arange(2003,2014)

PREFIX = r"/Volumes/My Passport/D/Data/ET/ALEXIET_annual/ET"
def readET_ts(PREFIX,rlist,clist):
    ET_ts = np.transpose(np.array([[] for i in range(len(rlist))]))
    tt = np.array([],dtype='datetime64')
    for year in yrange:
        print(year)
        ET_yr = np.load(PREFIX+str(year)+'.npy')  
        tmp = np.transpose(np.array([ET_yr[rlist[i],clist[i],:] for i in range(len(rlist))]))
        ET_ts = np.concatenate([ET_ts,tmp],axis=0)
        tt = np.concatenate([tt,np.arange(np.datetime64(str(year)+'-01-08'),np.datetime64(str(year+1)+'-01-01'), np.timedelta64(7,'D'))])
    return ET_ts,tt

ET_ts,tt = readET_ts(PREFIX,rlist,clist)

#%%
for i in range(len(rlist)):
    tmp_df = pd.DataFrame({'Time':tt,'ET':ET_ts[:,i]})
    tmp_df.to_csv('../ALEXI/ET_'+str(rlist[i])+'_'+str(clist[i])+'.csv')
    
#plt.plot(ET_ts[:,0]); plt.plot(ET_ts[:,1]); 

