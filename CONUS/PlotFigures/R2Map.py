#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 10:19:30 2020

@author: yanlan
"""

import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
import seaborn as sns; sns.set(style="ticks", color_codes=True,font_scale=1.75)
import os; os.environ['PROJ_LIB'] = '/Users/yanlan/opt/anaconda3/pkgs/proj4-5.2.0-h0a44026_1/share/proj/'
from mpl_toolkits.basemap import Basemap
import sys; sys.path.append("../Utilities/")
from Utilities import LatLon
from scipy.stats import norm,gamma
from newfun import get_var_bounds
from Utilities import LatLon


parentpath = '/Volumes/ELEMENTS/VOD_hydraulics/'
versionpath = parentpath + 'Retrieval_SM/';MODE = 'VOD_SM_ET'
inpath = parentpath+ 'Input/'
outpath = versionpath +'Output/'
forwardpath = versionpath+'Forward/'
statspath = versionpath+'STATS/'

MODE = 'VOD_SM_ET'
SiteInfo = pd.read_csv('../Utilities/SiteInfo_US_full.csv')

arrayid = 0
# sitename = str(SiteInfo['row'].values[fid])+'_'+str(SiteInfo['col'].values[fid])

N = 140
ACC_all = np.zeros([0,4])
PARA_all = np.zeros([0,14])
for arrayid in range(N):
    estname = statspath+'EST_'+MODE+'_'+str(arrayid).zfill(3)+'.pkl'
    if os.path.isfile(estname):
        with open(estname, 'rb') as f: 
            TS_mean,TS_std,PARA_mean,PARA_std,ACC = pickle.load(f)
        # VOD,ET,PSIL,S1 = TS_mean[sub_fid,:] # (TS_std[sub_fid,:]) temporal mean (std) of ensembel mean
        # g1,lpx,psi50X,C,bexp,bc,sigma_et,sigma_vod,loglik,a,b,c = PARA_mean[sub_fid,:] # (PARA_std[sub_fid,:]) ensemble mean (std)
        # r2_vod,r2_et,r2_sm,Geweke = ACC[sub_fid,:]
        ACC_all = np.concatenate([ACC_all,ACC],axis=0) 
        PARA_all = np.concatenate([PARA_all,PARA_mean],axis=0) 
    else:
        ACC_all = np.concatenate([ACC_all,np.zeros([100,4])+np.nan],axis=0) 
        PARA_all = np.concatenate([PARA_all,np.zeros([100,14])+np.nan],axis=0)
        

#%%
df = pd.DataFrame(ACC_all,columns=['R2_VOD','R2_ET','R2_SM','Geweke'])
df['P50'] = PARA_all[:,2]
df['row'] = SiteInfo['row'].iloc[:len(df)]
df['col'] = SiteInfo['col'].iloc[:len(df)]
df = df.dropna().reset_index()

lat,lon = LatLon(np.array(df['row']),np.array(df['col']))


#%%
varname = 'P50'; vmin = 0; vmax = 12

heatmap1_data = pd.pivot_table(df, values=varname, index='row', columns='col')
fig=plt.figure(figsize=(13.2,5))
m = Basemap(llcrnrlon = -128, llcrnrlat = 25, urcrnrlon = -62, urcrnrlat = 50)

m.drawcoastlines(linewidth=1)
m.drawcountries(linewidth=1)
m.drawstates(linewidth=1)
mycmap = sns.cubehelix_palette(rot=-.63, as_cmap=True)
# mycmap = sns.cubehelix_palette(6, rot=-.5, dark=.3,as_cmap=True)
cs = m.pcolormesh(np.unique(lon),np.flipud(np.unique(lat)),heatmap1_data,cmap=mycmap,vmin=vmin,vmax=vmax,shading='quad')
cbar = m.colorbar(cs)
cbar.set_label(varname,rotation=360,labelpad=15)
plt.show()

