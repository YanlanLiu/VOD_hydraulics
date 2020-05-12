#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 21:25:18 2020

@author: yanlan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
parentpath = '/Volumes/ELEMENTS/VOD_hydraulics/'
SiteInfo = pd.read_csv('SiteInfo_US_full.csv')
LAI = np.zeros([len(SiteInfo),2])+np.nan

#%%
for i in range(len(SiteInfo)):
    fname = parentpath+'/Input/LAI/LAI_'+str(SiteInfo['row'][i])+'_'+str(SiteInfo['col'][i])+'.csv'
    if os.path.exists(fname):
        tmp = np.array(pd.read_csv(fname)['Lai'])/10
        LAI[i,:] = np.array([np.percentile(tmp,50),np.percentile(tmp,75)])
SiteInfo['LAI_50'] = LAI[:,0]
SiteInfo['LAI_75'] = LAI[:,1]

#%%
heatmap1_data = pd.pivot_table(SiteInfo, values='LAI_75', index='row', columns='col')
plt.figure(figsize=(13.5,5))
plt.imshow(heatmap1_data,cmap='Greens'); plt.colorbar();plt.xticks([]);plt.yticks([])
plt.clim([0,4.2])
plt.title('LAI, 75th percentile')

