#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 20:50:37 2020

@author: yanlan
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 15:57:47 2020

@author: yanlan
"""

import os
import numpy as np
import pandas as pd
import sys; sys.path.append("../Utilities/")
from newfun import GetTrace
import pickle


parentpath = '/scratch/users/yanlan/'
arrayid = int(os.environ['SLURM_ARRAY_TASK_ID']) # 0-14
nsites_per_id = 1000


# parentpath = '/Volumes/ELEMENTS/VOD_hydraulics/'
#arrayid = 83
#nsites_per_id = 1

versionpath = parentpath + 'Retrieval_0510/'
inpath = parentpath+'Input/'
outpath = versionpath+'Output/'
forwardpath = versionpath+'Forward_test/'
traitpath = versionpath+'Traits/'
#forwardpath = versionpath+'Forward/'
#r2path = versionpath+'R2/'

SiteInfo = pd.read_csv('../Utilities/SiteInfo_US_full.csv')
MODE = 'AM_PM_ET_'

warmup, nsample = (0.8,100)
HSM = np.zeros([nsites_per_id,nsample])+np.nan


#%%
for i,fid in enumerate(range(arrayid*nsites_per_id,(arrayid+1)*nsites_per_id)):
    sitename = str(SiteInfo['row'][fid])+'_'+str(SiteInfo['col'][fid])
    tracename = outpath+MODE+sitename+'_'
    forwardname = forwardpath+MODE+sitename+'.pkl'
    if os.path.isfile(forwardname):
        with open(forwardname,'rb') as f:  # Python 3: open(..., 'rb')
            SVOD, SET, SPSIL,SPOPT = pickle.load(f)
        
        trace = GetTrace(tracename,0,optimal=False)
        trace = trace[trace['step']>trace['step'].max()*warmup].reset_index().drop(columns=['index'])
        p50 = -1*np.array([trace['psi50X'].iloc[max(len(trace)-1-count,0)] for count in range(nsample)])
        HSM[i,:] = np.nanpercentile(SPSIL,10,axis=1)-p50


#%%
hsmname = traitpath+'HSM_'+str(arrayid)+'_1E3.pkl'
with open(hsmname, 'wb') as f:
    pickle.dump(HSM, f)

