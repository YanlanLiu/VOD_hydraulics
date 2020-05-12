#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 20:48:23 2020

@author: yanlan
"""

import os
import numpy as np
import pandas as pd
import glob

#parentpath = '/Volumes/ELEMENTS/VOD_hydraulics/'
parentpath = '/scratch/users/yanlan/'
outpath = parentpath+'Output/'

SiteInfo = pd.read_csv('SiteInfo_US_full.csv')
MODE = 'AM_PM_ET_'
missing_row = np.array([])
for i in range(len(SiteInfo)):
    fname = outpath+MODE+str(SiteInfo['row'][i])+'_'+str(SiteInfo['col'][i])+'*.pickle'
    flist = glob.glob(fname)
    if len(flist)<5:
        missing_row = np.concatenate([missing_row,[SiteInfo['row'][i]]])
print(np.unique(missing_row))
         #print(str(i),SiteInfo['row'][i])
    
