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
Collection_PARA = np.zeros([len(SiteInfo),14])+np.nan

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
            Collection_PARA[subrange,:] = PARA_mean
            
# parentpath = '/Volumes/ELEMENTS/VOD_hydraulics/'

# nsites_per_id = 100
# versionpath = parentpath + 'Retrieval_SM/'
# statspath = versionpath+'STATS/'

# MODE = 'VOD_SM_ET'
# varnames, bounds = get_var_bounds(MODE)
# SiteInfo = pd.read_csv('SiteInfo_US_full.csv')
# Collection_ACC = np.zeros([len(SiteInfo),4])+np.nan
# Collection_PARA = np.zeros([len(SiteInfo),14])+np.nan
# #%%
# for arrayid in range(140):
#     if np.mod(arrayid,100)==0:print(arrayid)
#     fname = statspath+'EST_'+MODE+'_'+str(arrayid).zfill(3)+'.pkl'
#     subrange = np.arange(arrayid*nsites_per_id,min((arrayid+1)*nsites_per_id,len(SiteInfo)))
#     if os.path.isfile(fname):
#         with open(fname,'rb') as f:
#             TS_mean,TS_std,PARA_mean,PARA_std,ACC = pickle.load(f)
#         # print(PARA_mean.shape)
#         if ACC.shape[1]>0:
#             Collection_ACC[subrange,:] = ACC
#             Collection_PARA[subrange,:] = PARA_mean
mycmap = sns.cubehelix_palette(rot=-.63, as_cmap=True)
def plotmap(df,varname,vmin=0,vmax=1,cmap=mycmap):
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
    plt.show()
    return 0
#%%
df_para = pd.DataFrame(Collection_PARA,columns=varnames+['a','b','c'])
df_para['row'] = SiteInfo['row'];df_para['col'] = SiteInfo['col']
df_para = df_para.dropna()
plotmap(df_para,'psi50X',vmin=0,vmax=7,cmap='RdYlBu_r')



#%% PCA
from numpy import linalg
Y = np.array(df_para[varnames[:5]])
Y = Y[np.where(~np.isnan(Y[:,1]))[0],:]
Y[:,1] = Y[:,1]*Y[:,2]
# n, p = Y.shape
# C = (np.eye(n)-np.ones([n,n])/n)
# Yc = np.dot(C,Y)
Yc = (Y-np.nanmean(Y,axis=0))/np.nanstd(Y,axis=0)
S = np.dot(Yc.T,Yc)


w, v = linalg.eig(S)
sw = w[np.argsort(w)[::-1]]
sv = v[:,np.argsort(w)[::-1]]
f1 = np.dot(Yc,sv[:,0])
f2 = np.dot(Yc,sv[:,1])
plt.figure()
plt.plot(f1,f2,'o')

df_pca = df_para[['row','col']][:len(f1)]
df_pca['f1'] = f1
df_pca['f2'] = f2

#%%
import seaborn as sns; sns.set(style="ticks", color_codes=True,font_scale=1.75)
plotmap(df_pca,'f1',vmin=-2,vmax=2,cmap='RdYlBu_r')
plotmap(df_pca,'f2',vmin=-2,vmax=2,cmap='RdYlBu_r')

# heatmap1_data = pd.pivot_table(df_pca, values='f2', index='row', columns='col')
# mycmap = sns.cubehelix_palette(rot=-.63, as_cmap=True)

# plt.imshow(heatmap1_data,cmap='RdBu_r',vmin=-2,vmax=2);plt.colorbar()
# plt.xticks([]);plt.yticks([])
# plt.title('PC2')
#%%
IGBPlist = ['NA','ENF','EBF','DNF','DBF','MF','CSB','OSB',
            'WSV','SAV','GRA','CRO/URB','CRO','URB','GRA','SNW','NA','NA','NA']

plotmap(SiteInfo,'IGBP',vmin=0,vmax=13,cmap=mycmap)

#%%
plt.plot(np.cumsum(sw)/np.sum(sw)*100,'^-k')
plt.ylim([0,110])
plt.xlabel('number of PCs')
plt.ylabel('% of var. explained')
#%%
xlim = [-0.8,0.8]
plt.plot(sv[:,0],sv[:,1],'ok')
for i in range(5):
    plt.text(sv[i,0],sv[i,1],varnames[i])
plt.plot([0,0],xlim,'--k')
plt.plot(xlim,[0,0],'--k')
plt.xlim(xlim)
plt.ylim(xlim)
plt.xlabel('PC1')
plt.ylabel('PC2')

#%% K-means clustering
from sklearn.cluster import KMeans
import numpy as np
# X = np.array([[1, 2], [1, 4], [1, 0],
#               [10, 2], [10, 4], [10, 0]])
NCluster = 9
kmeans = KMeans(n_clusters=NCluster, random_state=0).fit(Yc)
kmeans.labels_
kmeans.cluster_centers_


#%%
import seaborn as sns; sns.set(style="ticks", color_codes=True,font_scale=1.15)

from mpl_toolkits.mplot3d import Axes3D
# colors = [sns.color_palette("BrBG_r",9)[i] for i in [1,3,5,7]]
colors = [sns.color_palette("BrBG_r",9)[i] for i in np.arange(9)]

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
for i in range(NCluster):
    # ax.scatter(kmeans.cluster_centers_[i,0],kmeans.cluster_centers_[i,1],kmeans.cluster_centers_[i,4],'o',s=50,label='C'+str(i))
    ax.scatter(kmeans.cluster_centers_[i,0],kmeans.cluster_centers_[i,2],kmeans.cluster_centers_[i,3],'o',color=colors[i],edgecolors='k',s=50,label='C'+str(i))
ax.set_xlabel('g1')
ax.set_ylabel('p50')
ax.set_zlabel('gpmax')
plt.legend(bbox_to_anchor=(1.35,1.05))
# for i in range(4):
#     ax.text(kmeans.cluster_centers_[i,0],kmeans.cluster_centers_[i,2],kmeans.cluster_centers_[:,3],'s')

#%%
import seaborn as sns; sns.set(style="ticks", color_codes=True,font_scale=1.75)

from matplotlib import colors
df_km =  df_para[['row','col']]
df_km['km'] = kmeans.labels_
# sns.color_palette("Set2")[3:8]
# cmap0 = colors.ListedColormap([sns.color_palette("colorblind")[i] for i in [0,2,4,6]])
# cmap0 = colors.ListedColormap([sns.color_palette("BrBG_r",9)[i] for i in [1,3,5,7]])
cmap0 = colors.ListedColormap([sns.color_palette("BrBG_r",9)[i] for i in np.arange(9)])

plotmap(df_km,'km',vmin=0,vmax=NCluster,cmap=cmap0)



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




