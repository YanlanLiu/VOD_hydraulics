#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 15:53:52 2020

@author: yanlan
"""

import os
import numpy as np
import pandas as pd
import glob
import sys; sys.path.append("../Utilities/")
from newfun import readCLM
import pickle
import time

parentpath = '/scratch/users/yanlan/' 
# arrayid = int(os.environ['SLURM_ARRAY_TASK_ID']) # 0-14
# nsites_per_id = 1000


# parentpath = '/Volumes/ELEMENTS/VOD_hydraulics/'
arrayid = 0
nsites_per_id = 2
inputpath = parentpath+'Input/'
n_path = inputpath+'ValidN/'

inpath = parentpath+'Input/'
SiteInfo = pd.read_csv('../Utilities/SiteInfo_US_full.csv')
MODE = 'AM_PM_ET_'

validN = np.zeros([len(SiteInfo),2])


for fid in range(arrayid*nsites_per_id,(arrayid+1)*nsites_per_id):
    sitename = str(SiteInfo['row'][fid])+'_'+str(SiteInfo['col'][fid])
    Forcings,VOD,ET,dLAI,discard_vod,discard_et,idx = readCLM(inpath,sitename)
    validN[fid,:] = np.array([np.sum(~np.isnan(VOD)),np.sum(~np.isnan(ET))])
    
n_name = n_path+'N_'+str(arrayid)+'_1E3.pkl'
np.save(n_name,validN)

