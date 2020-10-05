#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 13:25:07 2020

@author: yanlan
"""

import pickle
import os
import numpy as np
import pandas as pd
import warnings; warnings.simplefilter("ignore")
import sys; sys.path.append("../Utilities/")
from newfun import get_var_bounds
from newfun import GetTrace0
import time

tic = time.perf_counter()

# parentpath = '/scratch/users/yanlan/'
# arrayid = int(os.environ['SLURM_ARRAY_TASK_ID']) # 0-935
# nsites_per_id = 1000
# 
parentpath = '/Volumes/ELEMENTS/VOD_hydraulics/'
#arrayid = 10#4672
#nsites_per_id = 5
#warmup, nsample,thinning = (0.8,2,40)


numofchains = 3
versionpath = parentpath + 'Global_0817/'
inpath = parentpath+ 'Input_Global/'
outpath = versionpath +'Output/'
cpath = versionpath+'CVG/'

MODE = 'VOD_SM_ET'
varnames, bounds = get_var_bounds(MODE)
SiteInfo = pd.read_csv('SiteInfo_globe_full.csv')

#%%
for arrayid in range(0,10):
    estname = cpath+'CVG_'+str(arrayid).zfill(3)+'.pkl'
    with open(estname, 'rb') as f: CVG = pickle.load(f)
    print(CVG.shape)