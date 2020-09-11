#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 21:45:34 2020

@author: yanlan
"""

import pickle
import os
import numpy as np
import pandas as pd
import warnings; warnings.simplefilter("ignore")
import sys; sys.path.append("../Utilities/")
from newfun import readCLM # newfun_full
from newfun import fitVOD_RMSE,calVOD, dt, hour2day, hour2week
from newfun import get_var_bounds,OB,CONST,CLAPP,ca
from newfun import GetTrace
from Utilities import nanOLS,nancorr,MovAvg,calRMSE
import time
from scipy.stats import norm


tic = time.perf_counter()

# =========================== control pannel =============================

# parentpath = '/scratch/users/yanlan/'
# arrayid = int(os.environ['SLURM_ARRAY_TASK_ID']) # 0-935
# nsites_per_id = 1000

parentpath = '/Volumes/ELEMENTS/VOD_hydraulics/'
arrayid = 10#4672
nsites_per_id = 1

versionpath = parentpath + 'Global_0817/'
inpath = parentpath+ 'Input_Global/'
forwardpath = versionpath+'Forward/'
statspath = versionpath+'STATS_ACC/'


MODE = 'VOD_SM_ET'
varnames, bounds = get_var_bounds(MODE)
SiteInfo = pd.read_csv('SiteInfo_globe_full.csv')


ACC = []; ACCnan = [np.nan for i in range(12)]


idx_sigma_vod = varnames.index('sigma_vod')
idx_sigma_et = varnames.index('sigma_et')
idx_sigma_sm = varnames.index('sigma_sm')


for fid in range(arrayid*nsites_per_id,min((arrayid+1)*nsites_per_id,len(SiteInfo))):#range(953,954):#
    print(fid)
    sitename = str(SiteInfo['row'].values[fid])+'_'+str(SiteInfo['col'].values[fid])
    try:
        Forcings,VOD,SOILM,ET,dLAI,discard_vod,discard_et,idx = readCLM(inpath,sitename)
    except FileNotFoundError as err:
        print(err)
        ACC.append(ACCnan)
        continue
    RNET,TEMP,P,VPD,Psurf,GA,LAI,VegK = Forcings
    VOD_ma = np.reshape(VOD,[-1,2])
    VOD_ma = np.reshape(np.column_stack([MovAvg(VOD_ma[:,0],4),MovAvg(VOD_ma[:,1],4)]),[-1,])    
    
    # valid_vod = ~np.isnan(VOD_ma); VOD_ma_valid = VOD_ma[valid_vod]
    # valid_et = ~np.isnan(ET); ET_valid = ET[valid_et]
    # valid_sm = ~np.isnan(SOILM); SOILM_valid = SOILM[valid_sm]
    
    # calculate the three likelihoods

    forwardname = forwardpath+'TS_'+MODE+'_'+sitename+'.pkl'
    with open(forwardname, 'rb') as f: 
        TS = pickle.load(f)
        
    paraname = forwardpath+'PARA_'+MODE+'_'+sitename+'.pkl'
    with open(paraname, 'rb') as f: PARA = pickle.load(f)
    
    for sid in range(TS[0].shape[0]):
        VOD_hat, ET_hat, S1_hat = (TS[0][sid,:],TS[1][sid,:],TS[3][sid,:])
        sigma_VOD = PARA[1][sid,idx_sigma_vod]
        sigma_ET = PARA[1][sid,idx_sigma_et]
        sigma_SM = PARA[1][sid,idx_sigma_sm]
        loglik_vod = np.nanmean(norm.logpdf(VOD_ma,VOD_hat,sigma_VOD))
        loglik_et = np.nanmean(norm.logpdf(ET,ET_hat,sigma_ET))
        loglik_sm = np.nanmean(norm.logpdf(SOILM,S1_hat,sigma_SM))

    TS = [np.nanmean(itm,axis=0) for itm in TS] 
    
    er2 = [nancorr(TS[0],VOD_ma)**2, nancorr(TS[1],ET)**2, nancorr(TS[3],SOILM)**2] # VOD, ET, SM
    ermse = [calRMSE(VOD_ma,TS[0]), calRMSE(ET,TS[1]), calRMSE(SOILM,TS[3])]
    
    dVPD = hour2day(VPD,idx)[~discard_vod][1::2]
    wVPD = hour2week(VPD)[~discard_et]
    dry = (dVPD>np.nanpercentile(dVPD,75))
    ddry = np.repeat(dry,2) 
    wdry = (wVPD>np.nanpercentile(wVPD,75))
    
    er2_dry = [nancorr(TS[0][ddry],VOD_ma[ddry])**2, nancorr(TS[1][wdry],ET[wdry])**2, nancorr(TS[3][dry],SOILM[dry])**2]
    ermse_dry = [calRMSE(VOD_ma[ddry],TS[0][ddry]), calRMSE(ET[wdry],TS[1][wdry]), calRMSE(SOILM[dry],TS[3][dry])]

    acc_summary = er2+ermse+er2_dry+ermse_dry

    ACC.append(acc_summary)
ACC = np.array(ACC)

# estname = statspath+'FULL_'+str(arrayid).zfill(3)+'.pkl'
# with open(estname, 'wb') as f: 
#     pickle.dump(ACC, f)

toc = time.perf_counter()
    
    
print(f"Running time (100 sites): {toc-tic:0.4f} seconds")

