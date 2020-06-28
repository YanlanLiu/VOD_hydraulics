#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 16:03:16 2020

@author: yanlan
"""

from random import randint
import pickle
import os
import numpy as np
import pandas as pd
import warnings; warnings.simplefilter("ignore")
import sys; sys.path.append("../Utilities/")
from newfun import readCLM
from newfun import fitVOD_RMSE,dt, hour2day, hour2week
from newfun import OB,CONST,CLAPP,ca
from newfun import GetTrace
from Utilities import MovAvg,nanOLS
import time
import matplotlib.pyplot as plt
tic = time.perf_counter()

# =========================== control pannel =============================

# parentpath = '/scratch/users/yanlan/'
# arrayid = int(os.environ['SLURM_ARRAY_TASK_ID']) # 0-119
# nsites_per_id = 1
# warmup, nsample = (0.8,1000)

parentpath = '/Volumes/ELEMENTS/VOD_hydraulics/'
arrayid = 81
nsites_per_id = 1
warmup, nsample = (0.8,2)

versionpath = parentpath + 'TroubleShooting/Control/'
inpath = parentpath+ 'Input/'
# outpath = versionpath +'Output/'
outpath =  parentpath + 'Retrieval_0510/Output/'
forwardpath = versionpath+'Forward/'

MODE = 'AM_PM_ET_'
SiteInfo = pd.read_csv('SiteInfo_reps.csv')

# for fid in range(5,6):
fid = 5
sitename = str(SiteInfo['row'][fid])+'_'+str(SiteInfo['col'][fid])    
Forcings,VOD,SOILM,ET,dLAI,discard_vod,discard_et,idx = readCLM(inpath,sitename)

VOD_ma = np.reshape(VOD,[-1,2])
VOD_ma = np.reshape(np.column_stack([MovAvg(VOD_ma[:,0],4),MovAvg(VOD_ma[:,1],4)]),[-1,])

forwardname = forwardpath+MODE+sitename+'.pkl'
with open(forwardname,'rb') as f:
    SVOD, SE, ST, SPSIL, SS1, SS2, SPOPT, STHETA = pickle.load(f)
    
    
    
res = nanOLS(np.column_stack([VOD_ma[::2],dLAI[::2]]), VOD_ma[1::2])
print([res.params[0],1-res.conf_int(0.32)[0,0]/res.params[0]])

res = nanOLS(np.column_stack([SVOD[200,::2],dLAI[::2]]), SVOD[200,1::2])
print([res.params[0],1-res.conf_int(0.32)[0,0]/res.params[0]])
#%%   
# from scipy import stats
# def nanslope(x,y):
#     nanfilter = ~np.isnan(x+y)
#     return stats.linregress(x[nanfilter],y[nanfilter])
    
    
# from sklearn.linear_model import LinearRegression

# def nanreg(X,y):
#     nanfilter = ~np.isnan(np.sum(X,axis=1)+y)
#     reg = LinearRegression().fit(X[nanfilter,:],y[nanfilter])
#     return reg.coef_, reg.intercept_, reg.score(X[nanfilter,:],y[nanfilter])
    
# coef,intrecept,score = nanreg(np.column_stack([VOD_ma[::2],dLAI[::2]]), VOD_ma[1::2])


# nanreg(np.column_stack([SVOD[200,::2],dLAI[::2]]), SVOD[200,1::2])


#%%

# nsample = 100
# x = np.linspace(0, 10, nsample)
# X = np.column_stack((x, x**2))
# beta = np.array([1, 0.1, 10])
# e = np.random.normal(size=nsample)

# X = sm.add_constant(X)
# y = np.dot(X, beta) + e

# mod = sm.OLS(y, X)
# res = mod.fit()
# print(res.conf_int(0.01))
# res
# X.shape
#%%

# vod_ratio_obs = VOD[::2]/VOD[1::2]
# slope, intercept, r_value, p_value, std_err = nanslope(VOD_ma[::2],VOD_ma[1::2])
# print(slope,intercept, r_value, p_value, std_err)

# slope, intercept, r_value, p_value, std_err = nanslope(SVOD[200,::2],SVOD[300,1::2])
# print(slope,intercept, r_value, p_value, std_err)

# Add LAI to the regression? Slop and intercept is very different, refer to Konings and Gentine 2017 GCB

plt.plot(SVOD[200,::2]/SVOD[200,1::2])
plt.plot(VOD_ma[::2]/VOD_ma[1::2])
#%%
plt.plot(VOD[::2],VOD[1::2],'ok')
vod_ratio = SVOD[:,::2]/SVOD[:,1::2]
np.nanmean(np.nanstd(vod_ratio,axis=0))/np.nanmean(vod_ratio)
np.nanmean(np.nanstd(SVOD,axis=0))/np.nanmean(SVOD)
np.nanmean(np.nanstd(SE+ST,axis=0))/np.nanmean(SE+ST)
np.nanmean(np.nanstd(SPSIL,axis=0))/np.abs(np.nanmean(SPSIL))
np.nanmean(np.nanstd(SS1,axis=0))/np.abs(np.nanmean(SS1))    
np.nanmean(np.nanstd(SS2,axis=0))/np.abs(np.nanmean(SS2))

#%%
RNET,TEMP,P,VPD,Psurf,GA,LAI,VegK = Forcings
dVPD = hour2day(VPD,[idx[1]])
plt.plot(dVPD)

#%%
VOD_am = np.load('/Volumes/ELEMENTS/D/Data/VOD/AMSRE/Annual/VOD_am_2009.npy')
VOD_pm = np.load('/Volumes/ELEMENTS/D/Data/VOD/AMSRE/Annual/VOD_pm_2009.npy')

#%%
VOD_am[VOD_am<0] = np.nan
VOD_pm[VOD_pm<0] = np.nan
iso = np.nanmean(VOD_pm/VOD_am,axis=2)

#%%
plt.imshow(iso);plt.colorbar();plt.clim([0,1])
# plt.plot(SOILM)
    
    # [np.nanstd(np.nanmean(SVOD,axis=1)),np.nanmean(np.nanstd(SVOD,axis=1))]
    # [np.nanstd(np.nanmean(SE+ST,axis=1)),np.nanmean(np.nanstd(SE+ST,axis=1))]
    # [np.nanstd(np.nanmean(SPSIL,axis=1)),np.nanmean(np.nanstd(SPSIL,axis=1))]
    # [np.nanstd(np.nanmean(SS1,axis=1)),np.nanmean(np.nanstd(SS1,axis=1))]
    # [np.nanstd(np.nanmean(SS2,axis=1)),np.nanmean(np.nanstd(SS2,axis=1))]
#    forwardname = forwardpath+MODE+'_'+sitename+'.pkl'
#     with open(forwardname, 'wb') as f: 
#         pickle.dump([SVOD, SE, ST, SPSIL, SS1, SS2, SPOPT, STHETA], f) # add theta and S1
# with open(forwardname,'rb') as f:  # Python 3: open(..., 'rb')
#     SVOD, SE, ST, SPSIL, SS1, SS2, SPOPT, STHETA = pickle.load(f)

