#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 12:17:04 2020

@author: yanlan
"""
from random import randint
import pickle
import os
import numpy as np
import pandas as pd
import warnings; warnings.simplefilter("ignore")
import sys; sys.path.append("../Utilities/")
from newfun_global import readCLM # newfun_full
from newfun_global import fitVOD_RMSE,dt, hour2day, hour2week
from newfun_global import get_var_bounds,OB,CONST,CLAPP,ca
from newfun_global import GetTrace
from Utilities import nanOLS,nancorr
import time
from datetime import datetime

# parentpath = '/scratch/users/yanlan/Retrieval_VOD_ET/'


parentpath = '/Volumes/ELEMENTS/VOD_hydraulics/OSSE2/'
fid = 10


inpath0 = parentpath+ '../Input/'
inpath = parentpath+ '../Input_Global/'
SiteInfo = pd.read_csv('SiteInfo_reps_53.csv')

sitename = str(SiteInfo['row'].values[fid])+'_'+str(SiteInfo['col'].values[fid])
timerange = (datetime(2015,1,1), datetime(2017,1,1))

Forcings,VOD,SOILM,ET,dLAI,discard,amidx = readCLM(inpath,sitename,timerange)

#%%

# plt.plot(VOD)
# plt.plot(dLAI)
# idx_smap = np.where(tt_smap==tt_gldas[0].date())[0][0]
# amsr = pd.read_csv(inpath+'AMSRE/VOD_'+sitename+'.csv').drop(columns=['Unnamed: 0'])
# tt_amsr = np.arange(np.datetime64('2003-01-01'),np.datetime64('2012-01-01'))
# # tt_amsr = np.arange(np.datetime64(amsr['Time'][0]),np.datetime64(amsr['Time'][len(amsr)-1])+np.timedelta64(1,'D')).astype(datetime)
# idx_vod = np.where(tt_amsr==tt_gldas[0].date())[0][0]
# amsr = amsr[idx_vod:idx_vod+ndays]
# VOD = np.reshape(np.column_stack([rm_outlier(amsr['VOD_am']),rm_outlier(amsr['VOD_pm'])]),[-1,])[~discard_vod]
# SOILM = rm_outlier(amsr['SOILM_am'])[~discard_vod[::2]]/100

# alexi = pd.read_csv(inpath+'ALEXI/ET_'+sitename+'.csv')
# tt_alexi = np.array([itm for y in range(2003,2012) for itm in np.arange(np.datetime64(str(y)+'-01-08'),np.datetime64(str(y+1)+'-01-01'), np.timedelta64(7,'D'))])
# # tt_alexi = np.array([datetime.strptime(tmp,'%Y-%m-%d') for tmp in np.array(alexi['Time'])])
# ET = np.array(alexi['ET'])

# idx_et1 = np.where(tt_alexi>tt_gldas[0].date())[0][0]
# idx_et2 = np.where(tt_alexi<=tt_gldas[-1].date())[0][-1]+1

# if sum(np.isnan(ET))<len(ET)/2:
#     sn = np.nanmean(np.reshape(ET,[-1,52]),axis=0)  
#     ET = rm_outlier_sn(ET,sn)[idx_et1:idx_et2][~discard_et]
# else:
#     ET = ET[idx_et1:idx_et2][~discard_et]
    
# Obsv = (VOD,ET,dLAI)
# Discard = (discard_vod,discard_et,[amidx,pmidx])


# Forcings,VOD,SOILM,ET,dLAI,discard_vod,discard_et,idx = readCLM(inpath,sitename)
