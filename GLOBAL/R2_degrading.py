#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 20:53:26 2020

@author: yanlan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 14:33:23 2020

@author: yanlan
"""

import os
import numpy as np
import pandas as pd
import glob
from newfun import GetTrace, get_var_bounds
import pickle
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="ticks", color_codes=True,font_scale=1.75)
import os; os.environ['PROJ_LIB'] = '/Users/yanlan/opt/anaconda3/pkgs/proj4-5.2.0-h0a44026_1/share/proj/'
from mpl_toolkits.basemap import Basemap
from Utilities import LatLon
from matplotlib import cm

parentpath = '/Volumes/ELEMENTS/VOD_hydraulics/'

nsites_per_id = 100
versionpath = parentpath + 'Global_0817/'
statspath = versionpath+'STATS/'
# statspath = versionpath+'STATS_bkp/'

MODE = 'VOD_SM_ET'
varnames, bounds = get_var_bounds(MODE)
SiteInfo = pd.read_csv('SiteInfo_globe_full.csv')
Collection_ACC = np.zeros([len(SiteInfo),4])+np.nan

for arrayid in range(933):
    if np.mod(arrayid,100)==0:print(arrayid)
    fname = statspath+'EST_'+MODE+'_'+str(arrayid).zfill(3)+'.pkl'
    subrange = np.arange(arrayid*nsites_per_id,min((arrayid+1)*nsites_per_id,len(SiteInfo)))
    if os.path.isfile(fname):
        with open(fname,'rb') as f:
            TS_mean,TS_std,PARA_mean,PARA_std,ACC = pickle.load(f)
        # print(PARA_mean.shape)
        if ACC.shape[1]>0:
            Collection_ACC[subrange,:] = ACC

mycmap = sns.cubehelix_palette(rot=-.63, as_cmap=True)
def plotmap(df,varname,vmin=0,vmax=1,cmap=mycmap,title=''):
    heatmap1_data = pd.pivot_table(df, values=varname, index='row', columns='col')
    heatmap1_data[578] = np.nan
    plt.figure(figsize=(13.2,5))
    m = Basemap(llcrnrlon = -180.0, llcrnrlat = -60.0, urcrnrlon = 180.0, urcrnrlat = 90.0)
    m.drawcoastlines()
    m.drawcountries()
    lat,lon = LatLon(np.array(heatmap1_data.index),np.array(list(heatmap1_data)))
    cs = m.pcolormesh(lon,lat,heatmap1_data,cmap=cmap,vmin=vmin,vmax=vmax,shading='flat')
    cbar = m.colorbar(cs)
    cbar.set_label(varname,rotation=360,labelpad=15)
    plt.title(title)
    plt.show()
    return 0
#%%
df_para = pd.read_csv('SiteInfo_clusters.csv')
df_acc = pd.DataFrame(Collection_ACC,columns=['r2_vod','r2_et','r2_sm','Geweke'])
df_acc['row'] = SiteInfo['row'];df_acc['col'] = SiteInfo['col']
df_para = df_para.merge(df_acc,left_on=['row','col'],right_on=['row','col'],how='left')
plotmap(df_para,'r2_vod')



#%%
def getr2(prefix):

    for arrayid in range(66):
        fname = prefix+str(arrayid).zfill(3)+'.pkl'
        with open(fname,'rb') as f:
            tmp  = pickle.load(f)
        if arrayid==0:
            ACC = tmp.copy()
        else:
            ACC = np.concatenate([ACC,tmp],axis=0)
    
    df_pft = pd.DataFrame(ACC[:,:3],columns=['r2_vod','r2_et','r2_sm'])
    df_pft['row'] = df_para['row']; df_pft['col'] = df_para['col']
    return df_pft

df_pft = getr2(versionpath+'STATS_PFT/PFT_')
df_c4 = getr2(versionpath+'STATS_C4/Cluster_')
df_c12 =  getr2(versionpath+'STATS_C12/Cluster_')
df_full = df_para[['r2_vod','r2_et','r2_sm','row','col']]


degrad_pft = df_pft-df_full; degrad_pft['row']=df_para['row']; degrad_pft['col'] = df_para['col']
degrad_c4 = df_c12-df_full; degrad_c4['row']=df_para['row']; degrad_c4['col'] = df_para['col']
degrad_c12 = df_c12-df_full; degrad_c12['row']=df_para['row']; degrad_c12['col'] = df_para['col']

delta_c4 = df_c4-df_pft; delta_c4['row']=df_para['row']; delta_c4['col'] = df_para['col']
delta_c12 = df_c12-df_pft; delta_c12['row']=df_para['row']; delta_c12['col'] = df_para['col']

# plotmap(degrad_pft,'r2_vod',cmap='Reds_r',vmin=-0.25,vmax=0)

# #%%
# plt.plot(np.sort(delta_c12['r2_vod']),np.arange(len(delta_c4))/len(delta_c4))
# plt.xlim([-0.3,0.3])
# plt.plot([0,0],[0,1],'--k')
# plt.plot(np.sort(delta_c12['r2_vod']),np.arange(len(delta_c12))/len(delta_c12))

# print((delta_c12>0).sum()/len(delta_c12))

#%%
dd = 0.12; varname = 'r2_sm'
# plotmap(degrad_pft,varname,cmap='Reds_r',vmin=-2*dd,vmax=0,title="R2(pft)-R2")
# plotmap(delta_c12,varname,cmap='RdBu',vmin=-dd,vmax=dd,title="R2(clusters_12)-R2(pft)")

count,bins = np.histogram(delta_c4[varname],bins=[-1,-0.05,0,0.05,1])
cc = [sns.color_palette("RdBu",5)[i] for i in [0,1,3,4]]
plt.figure(figsize=(6,3))
for i in range(4):
    plt.bar(i,count[i]/len(delta_c4),color=cc[i])
plt.xticks(np.arange(4),['<-0.05','-0.05~0','0~0.05','>0.05'])
plt.xlabel(r'$\Delta R^2$')
plt.ylabel('fraction')
plt.ylim([0,0.5])



# plotmap(delta_c12,'r2_et',cmap='RdBu',vmin=-dd,vmax=dd)
# plotmap(delta_c12,'r2_sm',cmap='RdBu',vmin=-dd,vmax=dd)

# sns.kdeplot(degrad_c4['r2_vod'],bw=.02)
# sns.kdeplot(degrad_c4['r2_vod'],bw=.1)
# g = sns.PairGrid(delta_c4[['r2_vod','r2_et','r2_sm']])
# g.map_diag(sns.kdeplot)
# g.map_offdiag(sns.kdeplot, n_levels=6);

#%%

# from matplotlib import pyplot as plt
# import numpy as np

# ##generating some  data
# x,y = np.meshgrid(
#     np.linspace(0,1,100),
#     np.linspace(0,1,100),
# )
# directions = (np.sin(2*np.pi*x)*np.cos(2*np.pi*y)+1)*np.pi
# magnitude = np.exp(-(x*x+y*y))


# ##normalize data:
# def normalize(M):
#     return (M-np.min(M))/(np.max(M)-np.min(M))

# d_norm = normalize(directions)
# m_norm = normalize(magnitude)


# plt.imshow(d_norm);plt.colorbar()
# fig,(plot_ax, bar_ax) = plt.subplots(nrows=1,ncols=2,figsize=(8,4))

# plot_ax.imshow(
#     np.dstack((d_norm,m_norm, np.zeros_like(directions))),
#     cmap='seismic',
#     aspect = 'auto',
#     extent = (0,100,0,100),
# )

# bar_ax.imshow(
#     np.dstack((x, y, np.zeros_like(x))),
#     extent = (
#         np.min(directions),np.max(directions),
#         np.min(magnitude),np.max(magnitude),
#     ),
#     aspect = 'auto',
#     origin = 'lower',
# )
# bar_ax.set_xlabel('direction')
# bar_ax.set_ylabel('magnitude')

# plt.show()




