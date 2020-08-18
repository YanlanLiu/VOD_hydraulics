#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 21:58:08 2020

@author: yanlan
"""

import numpy as np
import pandas as pd

parentpath = '/Volumes/ELEMENTS/VOD_hydraulics/Input_Global/STATS/'

SiteInfo = pd.read_csv('SiteInfo_globe.csv')


for arrayid in range(114):
    tmp = pd.read_csv(parentpath+'Nobs_'+str(arrayid).zfill(3)+'.csv')
    if arrayid ==0:
        Nobs = tmp.copy()
    else:
        Nobs = pd.concat([Nobs,tmp],axis=0)

#%%

df = pd.DataFrame(np.concatenate([np.array(SiteInfo),np.array(Nobs)],axis=1),columns=['Unnamed: 0','row','col']+list(Nobs))

print(len(df))

df = df.loc[(df['f_ET']>0.5) & (df['f_VOD']>1/3) & ~((df['f_LAI']<1/3) & ~(np.isnan(df['f_LAI'])))]
print(len(df))
heatmap1_data = pd.pivot_table(df, values='f_ET', index='row', columns='col')
plt.imshow(heatmap1_data);plt.colorbar()

SiteInfo_short = pd.DataFrame(np.array(df[['row','col']]).astype(int),columns=['row','col'])
SiteInfo_short.to_csv('SiteInfo_glob_short.csv')
