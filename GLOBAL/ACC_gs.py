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
from newfun import readCLM_test # newfun_full
from newfun import fitVOD_RMSE,calVOD, dt, hour2day, hour2week
from newfun import get_var_bounds,OB,CONST,CLAPP,ca
from newfun import GetTrace
from Utilities import nanOLS,nancorr,MovAvg,calRMSE
import time
from scipy.stats import norm


tic = time.perf_counter()

# =========================== control pannel =============================

parentpath = '/scratch/users/yanlan/'
arrayid = int(os.environ['SLURM_ARRAY_TASK_ID']) # 0-935
nsites_per_id = 1000

# parentpath = '/Volumes/ELEMENTS/VOD_hydraulics/'
#arrayid = 10#4672
#nsites_per_id = 2

versionpath = parentpath + 'Global_0817/'
inpath = parentpath+ 'Input_Global/'
forwardpath = versionpath+'Forward/'
statspath = versionpath+'STATS_GS/'


MODE = 'VOD_SM_ET'
varnames, bounds = get_var_bounds(MODE)
SiteInfo = pd.read_csv('SiteInfo_globe_full.csv')


ACC = []; ACCnan = [np.nan for i in range(12)]


idx_sigma_vod = varnames.index('sigma_vod')
idx_sigma_et = varnames.index('sigma_et')
idx_sigma_sm = varnames.index('sigma_sm')

# from datetime import datetime, timedelta
# start_date = datetime(2003,7,2); end_date = datetime(2006,1,1)
# DOY = np.array([itm.timetuple().tm_yday for itm in [start_date+timedelta(days=i) for i in range((end_date-start_date).days)]])
accnan = [np.nan for i in range(12)]
for fid in range(arrayid*nsites_per_id,min((arrayid+1)*nsites_per_id,len(SiteInfo))):#range(953,954):#
    print(fid)
    sitename = str(SiteInfo['row'].values[fid])+'_'+str(SiteInfo['col'].values[fid])
    try:
        Forcings,VOD,SOILM,ET,dLAI,discard_vod,discard_et,idx,dLAI0 = readCLM_test(inpath,sitename)
    except (FileNotFoundError,KeyError) as err:
        print(err)
        ACC.append(ACCnan)
        continue
    RNET,TEMP,P,VPD,Psurf,GA,LAI,VegK = Forcings
    VOD_ma = np.reshape(VOD,[-1,2])
    VOD_ma = np.reshape(np.column_stack([MovAvg(VOD_ma[:,0],4),MovAvg(VOD_ma[:,1],4)]),[-1,])    


    forwardname = forwardpath+'TS_'+MODE+'_'+sitename+'.pkl'
    try:
        with open(forwardname, 'rb') as f: 
            TS = pickle.load(f)
    except:
        ACC.append(accnan)
        continue
        
    TS = [np.nanmean(itm,axis=0) for itm in TS] 
    
    er2 = [nancorr(TS[0],VOD_ma)**2, nancorr(TS[1],ET)**2, nancorr(TS[3],SOILM)**2] # VOD, ET, SM
    ermse = [calRMSE(VOD_ma,TS[0]), calRMSE(ET,TS[1]), calRMSE(SOILM,TS[3])]
    
    
    trd = np.median(dLAI0)
    gs_vod = (dLAI>trd)
    gs_sm = gs_vod[1::2]
    gs_et = (hour2week((np.repeat(dLAI0,4)>trd)*1,UNIT=1)>0.5)[~discard_et]

    
    er2_gs = [nancorr(TS[0][gs_vod],VOD_ma[gs_vod])**2, nancorr(TS[1][gs_et],ET[gs_et])**2, nancorr(TS[3][gs_sm],SOILM[gs_sm])**2]
    ermse_gs = [calRMSE(VOD_ma[gs_vod],TS[0][gs_vod]), calRMSE(ET[gs_et],TS[1][gs_et]), calRMSE(SOILM[gs_sm],TS[3][gs_sm])]

    acc_summary = er2+ermse+er2_gs+ermse_gs

    ACC.append(acc_summary)
ACC = np.array(ACC)

estname = statspath+'GS_'+str(arrayid).zfill(3)+'.pkl'
with open(estname, 'wb') as f: 
    pickle.dump(ACC, f)

toc = time.perf_counter()
    
    
print(f"Running time (100 sites): {toc-tic:0.4f} seconds")

#%%
a = np.random.normal(0,1,100)
print(np.nanmean(norm.logpdf(a,np.zeros(a.shape),1)))
