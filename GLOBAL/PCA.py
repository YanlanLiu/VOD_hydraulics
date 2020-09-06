#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 14:33:23 2020

@author: yanlan
"""

import os
import glob
import pickle
import numpy as np
from numpy import linalg
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import seaborn as sns; sns.set(style="ticks", color_codes=True,font_scale=1.75)
os.environ['PROJ_LIB'] = '/Users/yanlan/opt/anaconda3/pkgs/proj4-5.2.0-h0a44026_1/share/proj/'
from mpl_toolkits.basemap import Basemap
from newfun import GetTrace, get_var_bounds
from Utilities import LatLon


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
            
            # TS_mean,TS_std,PARA_mean,PARA_std,ACC = pickle.load(f)
            tmp = pickle.load(f)
            # TS_mean,TS_std,PARA_mean,PARA_std,PARA2_mean,PARA2_std,ACC = pickle.load(f)
        # print(PARA_mean.shape)
        if len(tmp)==5:
            TS_mean,TS_std,PARA_mean,PARA_std,ACC = tmp
        else:
            TS_mean,TS_std,PARA_mean,PARA_std,PARA2_mean,PARA2_std,ACC = tmp
            
        if ACC.shape[1]>0:
            Collection_ACC[subrange,:] = ACC
            Collection_PARA[subrange,:] = PARA_mean
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
df_para['row'] = SiteInfo['row'];df_para['col'] = SiteInfo['col']; df_para['IGBP'] = SiteInfo['IGBP']
df_para['Root depth'] = SiteInfo['Root depth']; df_para['Soil texture'] = SiteInfo['Soil texture']
df_para['Vcmax25'] = SiteInfo['Vcmax25']
plotmap(df_para,'psi50X',vmin=0,vmax=7,cmap='RdYlBu_r')
df_para = df_para.dropna()

Y = np.array(df_para[varnames[:5]])
Y[:,2] = -Y[:,2] # P50X
Y[:,1] = Y[:,1]*Y[:,2] # P50S
Ymean = np.nanmean(Y,axis=0)
Ystd = np.nanstd(Y,axis=0)
Yc = (Y-Ymean)/Ystd
    
#%% K-means clustering

def fit_kmeans(Yc,n_clusters):
    R,K = Yc.shape
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(Yc)
    
    SST = 0
    Yhat = Yc.copy()
    for n in range(n_clusters):
        Yn = Yc[kmeans.labels_==n,:]
        mun = np.mean(Yn,axis=0)
        SST = SST+np.sum((Yn-mun)**2)
        Yhat[kmeans.labels_==n,:] = kmeans.cluster_centers_[n,:]
    sigma2 = SST/K/(R-K)
    Vall = np.sum((Yc-np.mean(Yc,axis=0))**2)/R
    Ve =  np.sum((Yhat-np.mean(Yhat,axis=0))**2)/R
    Vb = np.sum((kmeans.cluster_centers_-np.mean(kmeans.cluster_centers_,axis=0))**2)/n_clusters
    Vw = SST/R
    
    loglik = 0
    for n in range(n_clusters):
        Yn = Yc[kmeans.labels_==n,:]
        Rn = Yn.shape[0]
        mun = np.mean(Yn,axis=0)
        # loglik = loglik+np.sum(np.log(multivariate_normal.pdf(Yn, mean=mun, cov=sigma2*np.eye(K))))+Rn*np.log(Rn/R)
        loglik = loglik+np.sum(np.log(multivariate_normal.pdf(Yn, mean=mun, cov=sigma2*np.eye(K))*Rn/R))
    BIC = (K*n_clusters+1)*np.log(R)-2*loglik
    # AIC = 2*(K*n_clusters+1)-2*loglik
    return kmeans.labels_, kmeans.cluster_centers_,Vall,Ve,Vb,Vw,BIC

#%% A figure showing decay of BIC and Vw/Vb as n_clusters increases

# SVe = []; SVb = []; SVw = []; SBIC = []
# for n_clusters in range(3,20):
#     print(n_clusters)
#     klabel, kcenter, Vall,Ve,Vb,Vw, BIC = fit_kmeans(Yc,n_clusters)
#     SVe.append(Ve); SVb.append(Vb); SVw.append(Vw); SBIC.append(BIC)
# plt.figure()
# plt.plot(np.arange(3,20),np.array(SVe)/Vall,'-b',label='Var(explained)/Var(all)')
# # plt.plot(12,Vb_pft/Vall,'^b')
# plt.plot(np.arange(3,20),np.array(SVw)/np.array(SVb),'-r',label='Var(within)/Var(across)')
# plt.plot([12,12],[0.2,1],'--k',label='# of PFT')
# plt.legend(bbox_to_anchor=[1.05,1.05])
# plt.xlabel('# of clusters')
# print(Ve_pft/Vall,Vw_pft/Vb_pft)

# plt.figure()
# plt.plot(np.arange(3,20),SBIC)
# plt.ylabel('BIC')
# plt.xlabel('# of clusters')


#%%
n_clusters = 6
klabel, kcenter, Vall,Ve,Vb,Vw, BIC = fit_kmeans(Yc,n_clusters)
print(Ve/Vall,Vw/Vb,BIC)
sorted_idx = list(np.argsort(kcenter[:,2]))
kcenter_s = kcenter[sorted_idx,:]
klabel_s = np.array([sorted_idx.index(i) for i in klabel])


#%% 
cc = [sns.color_palette("Paired")[i] for i in [3,0,1,9,8]]
dd = 0.15
vnames = [r'$g_1$',r'$\psi_{50,s}$',r'$\psi_{50,x}$',r'$g_{p,max}$',r'$C$']
plt.figure(figsize=(10,5))
vid = 0
plt.bar(np.arange(n_clusters)+(vid-2)*dd,kcenter_s[:,vid],width=dd,color=cc[vid],label=vnames[vid])
plt.bar(1,2,color='w',alpha=1,label=' ')
for vid in [2,1,3,4]:
    plt.bar(np.arange(n_clusters)+(vid-2)*dd,kcenter_s[:,vid],width=dd,color=cc[vid],label=vnames[vid])
plt.legend(ncol=3)
plt.xticks(np.arange(n_clusters),['C'+str(i) for i in range(n_clusters)])
plt.ylabel('Normalized value')


#%%
df_km =  df_para[['row','col']]
df_km['clusters'] = klabel_s
cc = [sns.color_palette("Paired")[i] for i in [1,0,3,2,11,6]] # [0,1,2,3,6,11]
cmap0 = colors.ListedColormap(cc)
# plotmap(df_km,'clusters',vmin=0,vmax=n_clusters,cmap=cmap0)
heatmap1_data = pd.pivot_table(df_km, values='clusters', index='row', columns='col')
heatmap1_data[578] = np.nan
plt.figure(figsize=(13.2,5))
m = Basemap(llcrnrlon = -180.0, llcrnrlat = -60.0, urcrnrlon = 180.0, urcrnrlat = 90.0)
m.drawcoastlines()
m.drawcountries()
lat,lon = LatLon(np.array(heatmap1_data.index),np.array(list(heatmap1_data)))
cs = m.pcolormesh(lon,lat,heatmap1_data,cmap=cmap0,vmin=-0.5,vmax=5.5,shading='flat')
cbar = m.colorbar(cs)
cbar.ax.set_yticklabels(['C'+str(i) for i in range(n_clusters)])
# cbar.set_label(varname,rotation=360,labelpad=15)
plt.show()


#%% PCA
S = np.dot(Yc.T,Yc)
w, v = linalg.eig(S)
sw = w[np.argsort(w)[::-1]]
sv = v[:,np.argsort(w)[::-1]]
f1 = np.dot(Yc,sv[:,0])
f2 = np.dot(Yc,sv[:,1])
# plt.figure()
# plt.plot(f1,f2,'o')
df_pca = df_para[['row','col']][:len(f1)]
df_pca['f1'] = f1
df_pca['f2'] = f2

# import seaborn as sns; sns.set(style="ticks", color_codes=True,font_scale=1.75)
# plotmap(df_pca,'f1',vmin=-2,vmax=2,cmap='RdYlBu_r')
# plotmap(df_pca,'f2',vmin=-2,vmax=2,cmap='RdYlBu_r')
plt.plot(np.cumsum(sw)/np.sum(sw)*100,'^-k')
plt.ylim([0,110])
plt.xlabel('number of PCs')
plt.ylabel('% of var. explained')

plt.figure()
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


IGBPlist = ['NA','ENF','EBF','DNF','DBF','MF','CSB','OSB',
            'WSV','SAV','GRA','CRO/URB','CRO','URB','GRA','SNW','NA','NA','NA']

plotmap(df_para,'IGBP',vmin=0,vmax=13,cmap=mycmap)


#% variance within each PFT
IGBParray = np.array(df_para['IGBP'])
IGBPunique = np.unique(IGBParray)
count = [sum(df_para['IGBP']==itm) for itm in IGBPunique]
SST = 0
PFTcenter = []
Yhat = Yc.copy()
for itm in IGBPunique:
    Yn = Yc[IGBParray==itm,:]
    mun = np.mean(Yn,axis=0)
    PFTcenter.append(mun)
    SST = SST+np.sum((Yn-mun)**2)
    Yhat[IGBParray==itm,:] = mun
Vw_pft =  SST/Yc.shape[0]

PFTcenter = np.array(PFTcenter)
Vb_pft = np.sum((PFTcenter-np.mean(PFTcenter,axis=0))**2)/len(IGBPunique)
Ve_pft =  np.sum((Yhat-np.mean(Yhat,axis=0))**2)/len(Yc)



#%% BKP code

# colors = [sns.color_palette("BrBG_r",9)[i] for i in [1,3,5,7]]
# cc = [sns.color_palette("BrBG",n_clusters)[i] for i in np.arange(n_clusters)]

# fig = plt.figure(figsize=(6,6))
# ax = fig.add_subplot(111, projection='3d')
# for i in range(n_clusters):
#     # ax.scatter(kmeans.cluster_centers_[i,0],kmeans.cluster_centers_[i,1],kmeans.cluster_centers_[i,4],'o',s=50,label='C'+str(i))
#     ax.scatter(kcenter_s[i,2],kcenter_s[i,3],kcenter_s[i,0],'o',color=cc[i],edgecolors='k',s=50,label='C'+str(i))
# ax.set_xlabel('p50')
# ax.set_ylabel('gpmax')
# ax.set_zlabel('g1')
# plt.legend(bbox_to_anchor=(1.35,1.05))

