#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 11:06:15 2020

@author: yanlan
"""

import numpy as np
import os
import pandas as pd

LAIpath = '/Volumes/ELEMENTS/VOD_hydraulics/Input_Global/LAI/'

SiteInfo = pd.read_csv('../Traits/SiteInfo_globe.csv')
missingsites = []
for i in range(len(SiteInfo)):
    if np.mod(i,1000)==0: print(i,len(SiteInfo))
    sitename = str(SiteInfo['row'].iloc[i])+'_'+str(SiteInfo['col'].iloc[i])
    if os.path.isfile(LAIpath+'LAI_'+sitename+'.csv')==False:
        missingsites.append(SiteInfo.iloc[i].values)

#%%
missingsites = pd.DataFrame(np.array(missingsites),columns=list(SiteInfo))
missingsites.to_csv('MissingSites.csv')
