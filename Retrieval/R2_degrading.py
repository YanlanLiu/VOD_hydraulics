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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from Utilities import LatLon
from matplotlib import cm

parentpath = '/Volumes/ELEMENTS/VOD_hydraulics/'

nsites_per_id = 100
versionpath = parentpath + 'Global_0817/'
# statspath = versionpath+'STATS/'
statspath = versionpath+'STATS_bkp/STATS/'

MODE = 'VOD_SM_ET'
varnames, bounds = get_var_bounds(MODE)
SiteInfo = pd.read_csv('SiteInfo_globe_full.csv')
Collection_ACC = np.zeros([len(SiteInfo),4])+np.nan
Collection_PARA = np.zeros([len(SiteInfo),17])+np.nan
# Collection_S = np.zeros([len(SiteInfo),14])+np.nan

Collection_OBS = np.zeros([len(SiteInfo),9])+np.nan
Collection_N = np.zeros([len(SiteInfo),3])+np.nan

for arrayid in range(933):
    if np.mod(arrayid,100)==0:print(arrayid)
    subrange = np.arange(arrayid*nsites_per_id,min((arrayid+1)*nsites_per_id,len(SiteInfo)))
    fname = statspath+'EST_'+MODE+'_'+str(arrayid).zfill(3)+'.pkl'
    if os.path.isfile(fname):
        with open(fname,'rb') as f:
            TS_mean,TS_std,PARA_mean,PARA_std,PARA2_mean,PARA2_std,ACC = pickle.load(f)
        if ACC.shape[1]>0:
            Collection_ACC[subrange,:] = ACC
            Collection_PARA[subrange,:] = PARA_mean
            
mycmap = sns.cubehelix_palette(rot=-.63, as_cmap=True)
def plotmap(df,varname,vmin=0,vmax=1,cmap=mycmap,title='',cbartitle=''):
    heatmap1_data = pd.pivot_table(df, values=varname, index='row', columns='col')
    heatmap1_data[578] = np.nan
    plt.figure(figsize=(13.2,5))
    m = Basemap(llcrnrlon = -180.0, llcrnrlat = -60.0, urcrnrlon = 180.0, urcrnrlat = 90.0)
    m.drawcoastlines()
    m.drawcountries()
    lat,lon = LatLon(np.array(heatmap1_data.index),np.array(list(heatmap1_data)))
    cs = m.pcolormesh(lon,lat,heatmap1_data,cmap=cmap,vmin=vmin,vmax=vmax,shading='flat')
    cbar = m.colorbar(cs)
    cbar.set_label(cbartitle,rotation=360,labelpad=15)
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

    for arrayid in range(88):
        fname = prefix+str(arrayid).zfill(3)+'.pkl'
        with open(fname,'rb') as f:
            tmp  = pickle.load(f)
        if arrayid==0:
            ACC = tmp.copy()
        else:
            ACC = np.concatenate([ACC,tmp],axis=0)
    
    df_pft = pd.DataFrame(ACC[:,6:],columns=['r2_vod','r2_et','r2_sm','rmse_vod','rmse_et','rmse_sm'])
    df_pft['row'] = df_para['row']; df_pft['col'] = df_para['col']
    return df_pft

df_full = getr2(versionpath+'STATS_ACC/FULL_')
df_pft = getr2(versionpath+'STATS_PFT/PFT_')
df_c4 = getr2(versionpath+'STATS_C6/Cluster_')
# df_full = df_para[['r2_vod','r2_et','r2_sm','row','col']]


degrad_pft = df_pft-df_full; degrad_pft['row']=df_para['row']; degrad_pft['col'] = df_para['col']
degrad_c4 = df_c4-df_full; degrad_c4['row']=df_para['row']; degrad_c4['col'] = df_para['col']

delta_c4 = df_pft-df_c4; delta_c4['row']=df_para['row']; delta_c4['col'] = df_para['col']

print(degrad_c4.median())
print(degrad_pft.median())

print(delta_c4.median())

# plotmap(degrad_pft,'r2_vod',cmap='Reds_r',vmin=-0.5,vmax=0)
# plotmap(degrad_c4,'r2_vod',cmap='Reds_r',vmin=-0.5,vmax=0)

# #%%
# plt.plot(np.sort(delta_c12['r2_vod']),np.arange(len(delta_c4))/len(delta_c4))
# plt.xlim([-0.3,0.3])
# plt.plot([0,0],[0,1],'--k')
# plt.plot(np.sort(delta_c12['r2_vod']),np.arange(len(delta_c12))/len(delta_c12))

# print((delta_c12>0).sum()/len(delta_c12))


#%%
def plotdelta_map(df,varname,vmin=0,vmax=1,cmap=mycmap,cbartitle='',inset=True,bp=[0.1,0.2],borderpad=3,drawmask=True):
    heatmap1_data = pd.pivot_table(df, values=varname, index='row', columns='col')
    heatmap1_data[578] = np.nan
    fig, ax = plt.subplots(figsize=(13.2,5))
    m = Basemap(llcrnrlon = -180.0, llcrnrlat = -80, urcrnrlon = 180.0, urcrnrlat = 90.0)
    if drawmask: m.drawlsmask(land_color=(0.87,0.87,0.87),lakes=True)

    m.drawcountries()
    lcol = m.drawcoastlines()
    segs = lcol.get_segments()
    for i, seg in enumerate(segs):
      # The segments are lists of ordered pairs in spherical (lon, lat), coordinates.
      # We can filter out which ones correspond to Antarctica based on latitude using numpy.any()
        if np.any(seg[:, 1] < -60):
            segs.pop(i)
    lcol.set_segments(segs)
    

    lat,lon = LatLon(np.array(heatmap1_data.index),np.array(list(heatmap1_data)))
    cs = m.pcolormesh(lon,lat,heatmap1_data,cmap=cmap,vmin=vmin,vmax=vmax,shading='flat')
    cbar = m.colorbar(cs)
    if inset:
        bins = [-np.inf,-bp[1],-bp[0],bp[0],bp[1],np.inf]
        xticklabels = ['<'+str(-bp[1]),'('+str(-bp[1])+','+str(-bp[0])+')','('+str(-bp[0])+','+str(bp[0])+')','('+str(bp[0])+','+str(bp[1])+')','>'+str(bp[1])]
        # axins = inset_axes(ax, "15%", "20%" ,bbox_to_anchor=(-285,-100,600,300)) #-80
        # axins = inset_axes(ax, "15%", "20%" ,bbox_to_anchor=(-360,-108,650,300)) -65
        # axins = inset_axes(ax, "25%", "30%" ,bbox_to_anchor=(-140,-23,600,300)) #-80

        axins = inset_axes(ax, "50%", "30%" ,bbox_to_anchor=(300,0,600,300)) #-80
        plt.setp(axins.get_xticklabels(), fontsize=14,rotation=90)
        plt.setp(axins.get_yticklabels(), fontsize=14)
        count,bins = np.histogram(df[varname],bins=bins)

        cc = [sns.color_palette("RdBu_r",5)[i] for i in [0,1,2,3,4]]
        for i,itm in enumerate([0,1,2,3,4]):
            axins.bar(i,count[itm]/len(df),color=cc[i],edgecolor='k')
        plt.xticks(np.arange(5),xticklabels)
        plt.ylim(0,0.5)
        # plt.xticks([])


dd = 1.25; bp = [0.1, 0.5]#[-np.inf,-0.2,-0.1,0,0.1,0.2,np.inf]

plotdelta_map(delta_c4,'rmse_vod',cmap="RdBu_r",vmin=-dd,vmax=dd,bp=bp,inset=False)#,drawmask=False)
plt.savefig('../Figures/Fig8a_drmse_vod.png',dpi=300,bbox_inches='tight')

plotdelta_map(delta_c4,'rmse_et',cmap='RdBu_r',vmin=-dd,vmax=dd,bp=bp,inset=False)#,drawmask=False)
plt.savefig('../Figures/Fig8b_drmse_et.png',dpi=300,bbox_inches='tight')

#%%

bins = [-np.inf,-bp[1],-bp[0],bp[0],bp[1],np.inf]
xticklabels = ['<'+str(-bp[1]),'('+str(-bp[1])+','+str(-bp[0])+')','('+str(-bp[0])+','+str(bp[0])+')','('+str(bp[0])+','+str(bp[1])+')','>'+str(bp[1])]
plt.figure(figsize=(3,1.5))

varname ='rmse_vod'
count,bins = np.histogram(delta_c4[varname],bins=bins)

cc = [sns.color_palette("RdBu_r",5)[i] for i in [0,1,2,3,4]]
for i,itm in enumerate([0,1,2,3,4]):
    plt.bar(i,count[itm]/len(delta_c4),color=cc[i],edgecolor='k')
plt.xticks(np.arange(5),xticklabels,rotation=90)
plt.ylim(0,0.5)
plt.yticks([0,0.5])
plt.savefig('../Figures/Fig8b_drmse_vod_inset.png',dpi=300,bbox_inches='tight')

#%%
plt.figure(figsize=(3,1.5))

varname ='rmse_et'
count,bins = np.histogram(delta_c4[varname],bins=bins)

cc = [sns.color_palette("RdBu_r",5)[i] for i in [0,1,2,3,4]]
for i,itm in enumerate([0,1,2,3,4]):
    plt.bar(i,count[itm]/len(delta_c4),color=cc[i],edgecolor='k')
plt.xticks(np.arange(5),xticklabels,rotation=90)
plt.ylim(0,0.5)
plt.yticks([0,0.5])
plt.savefig('../Figures/Fig8b_drmse_et_inset.png',dpi=300,bbox_inches='tight')


#%%
def getr2(prefix):
    nsites_per_id = 1000

    for arrayid in range(int(len(SiteInfo)/nsites_per_id)+1):
        fname = prefix+str(arrayid).zfill(3)+'.pkl'
        if os.path.isfile(fname):
           with open(fname,'rb') as f:
               tmp  = pickle.load(f)
        if arrayid==0:
            ACC = tmp.copy()
        else:
            ACC = np.concatenate([ACC,tmp],axis=0)
    
    df_pft = pd.DataFrame(ACC[:,6:],columns=['r2_vod','r2_et','r2_sm','rmse_vod','rmse_et','rmse_sm'])
    df_pft['row'] = df_para['row']; df_pft['col'] = df_para['col']
    return df_pft

df_gs = getr2(versionpath+'STATS_GS/GS_')
plotmap(df_gs,'r2_sm',vmin=0,vmax=1)

# plotmap(df_gs,'r2_et',vmin=0,vmax=1)


#%%
# plotmap(degrad_c4,'rmse_vod',cmap='RdBu_r',vmin=-2,vmax=2,title="RMSE(hft)-RMSE, VOD, dry")
# plotmap(degrad_c4,'rmse_et',cmap='RdBu_r',vmin=-.5,vmax=.5,title="RMSE(hft)-RMSE(pft), ET, dry")
# plotmap(degrad_c4,'rmse_sm',cmap='RdBu_r',vmin=-.5,vmax=.5,title="RMSE(hft)-RMSE(pft), SM, dry")

# plotmap(degrad_c4,'r2_vod',cmap='RdBu',vmin=-.5,vmax=.5,title="R2(hft)-R2, VOD")
# plotmap(degrad_c4,'r2_et',cmap='RdBu',vmin=-.5,vmax=.5,title="R2(hft)-R2, ET")
# plotmap(degrad_c4,'r2_sm',cmap='RdBu',vmin=-.5,vmax=.5,title="R2(hft)-R2, SM")

# plotmap(degrad_pft,'r2_vod',cmap='RdBu',vmin=-.5,vmax=.5,title="R2(pft)-R2, VOD")
# plotmap(degrad_pft,'r2_et',cmap='RdBu',vmin=-.5,vmax=.5,title="R2(pft)-R2, ET")
# plotmap(degrad_pft,'r2_sm',cmap='RdBu',vmin=-.5,vmax=.5,title="R2(pft)-R2, SM")

dd = 2.8
plotmap(degrad_c4,'rmse_vod',cmap='RdBu_r',vmin=-dd,vmax=dd,title="RMSE(hft)-RMSE, VOD")
plotmap(degrad_c4,'rmse_et',cmap='RdBu_r',vmin=-dd,vmax=dd,title="RMSE(hft)-RMSE, ET")
# plotmap(degrad_c4,'rmse_sm',cmap='RdBu_r',vmin=-dd,vmax=dd,title="RMSE(hft)-RMSE, SM")


dd = 1
plotmap(delta_c4,'rmse_vod',cmap='RdBu_r',vmin=-dd,vmax=dd,title="RMSE(hft)-RMSE, VOD")
plotmap(delta_c4,'rmse_et',cmap='RdBu_r',vmin=-dd,vmax=dd,title="RMSE(hft)-RMSE, ET")

# plotmap(delta_c4,'rmse_vod',cmap='RdBu_r',vmin=-2,vmax=2,title="RMSE(hft)-RMSE(pft), VOD")
# plotmap(delta_c4,'rmse_et',cmap='RdBu_r',vmin=-.5,vmax=.5,title="RMSE(hft)-RMSE(pft), ET")

# plotmap(degrad_pft,'rmse_vod',cmap='RdBu',vmin=-.5,vmax=.5,title="R2(pft)-R2, VOD")
# plotmap(degrad_pft,'rmse_et',cmap='RdBu',vmin=-.5,vmax=.5,title="R2(pft)-R2, ET")
# plotmap(degrad_pft,'rmse',cmap='RdBu',vmin=-.5,vmax=.5,title="R2(pft)-R2, SM")



#%%
def plotdegrade_map(df,varname,vmin=0,vmax=1,cmap=mycmap,cbartitle='',inset=True,borderpad=2.4,drawmask=True):
    heatmap1_data = pd.pivot_table(df, values=varname, index='row', columns='col')
    heatmap1_data[578] = np.nan
    fig, ax = plt.subplots(figsize=(13.2,5))
    m = Basemap(llcrnrlon = -180.0, llcrnrlat = -60.0, urcrnrlon = 180.0, urcrnrlat = 90.0)
    if drawmask: m.drawlsmask(land_color=(0.87,0.87,0.87),lakes=True)
    m.drawcoastlines()
    m.drawcountries()
    lat,lon = LatLon(np.array(heatmap1_data.index),np.array(list(heatmap1_data)))
    cs = m.pcolormesh(lon,lat,heatmap1_data,cmap=cmap,vmin=vmin,vmax=vmax,shading='flat')
    cbar = m.colorbar(cs)
       
dd = 2.8
plotdegrade_map(degrad_c4,'rmse_vod',cmap='RdBu_r',vmin=-dd,vmax=dd)
plotdegrade_map(degrad_c4,'rmse_et',cmap='RdBu_r',vmin=-dd,vmax=dd)

#%%
def plotdelta_map(df,varname,vmin=0,vmax=1,cmap=mycmap,cbartitle='',inset=True,bp=[0.1,0.2],borderpad=2.2,drawmask=True):
    heatmap1_data = pd.pivot_table(df, values=varname, index='row', columns='col')
    heatmap1_data[578] = np.nan
    fig, ax = plt.subplots(figsize=(13.2,5))
    m = Basemap(llcrnrlon = -180.0, llcrnrlat = -80.0, urcrnrlon = 180.0, urcrnrlat = 90.0)
    if drawmask: m.drawlsmask(land_color=(0.87,0.87,0.87),lakes=True)
    m.drawcoastlines()
    m.drawcountries()
    lat,lon = LatLon(np.array(heatmap1_data.index),np.array(list(heatmap1_data)))
    cs = m.pcolormesh(lon,lat,heatmap1_data,cmap=cmap,vmin=vmin,vmax=vmax,shading='flat')
    cbar = m.colorbar(cs)
    if inset:
        bins = [-np.inf,-bp[1],-bp[0],0,bp[0],bp[1],np.inf]
        ['<-0.2','-0.2~-0.1','0.1~0.2','>0.2']
        xticklabels = ['<'+str(-bp[1]),str(-bp[1])+'~'+str(-bp[0]),str(bp[0])+'~'+str(bp[1]),'>'+str(bp[1])]
        axins = inset_axes(ax,  "18%", "30%" ,loc="lower left", borderpad=borderpad)
        plt.setp(axins.get_xticklabels(), fontsize=16,rotation=30)
        plt.setp(axins.get_yticklabels(), fontsize=16)
        count,bins = np.histogram(df[varname],bins=bins)

        cc = [sns.color_palette("RdBu_r",5)[i] for i in [0,1,3,4]]
        for i,itm in enumerate([0,1,4,5]):
            axins.bar(i,count[itm]/len(df),color=cc[i])
        # plt.xticks(np.arange(4),xticklabels)
        plt.xticks([])
        # plt.xlabel(r'$\Delta RMSE$')
        # plt.ylabel('fraction')

        # sns.kdeplot(df[varname],ax=axins,legend=False,color=c1)
        # sns.kdeplot(df_std[varname][df_std[varname]<3],cut=0,bw=0.025,color='k',legend=False)
        # plt.xlim([-0.1,1.52])
        # plt.xticks([0,0.5,1])
        
        # xlim = axins.get_xlim()
        # ylim = axins.get_ylim()
        # # axins.plot(df[varname].median()*np.array([1,1]),[0,ylim[1]],'--',color=c1)
        # axins.text(xlim[0]-(xlim[1]-xlim[0])/4,ylim[1]+(ylim[1]-ylim[0])/5,'Area fraction',fontsize=16)
        # axins.text(-0.5,ylim[1]+(ylim[1]-ylim[0])/20,'pdf',fontsize=18)
        # axins.text(0,ylim[0]-(ylim[1]-ylim[0])/2,cbartitle,fontsize=16)
     
# dd = 1.25; bp = [0.25, 0.75]#[-np.inf,-0.2,-0.1,0,0.1,0.2,np.inf]

# colors = [sns.color_palete("RdBu_r",5)[i] for i in range(5)]
# cmap = cmap_discretize(sns.color_palette("RdBu_r",5,as_cmap=True), n_colors=5)

dd = 1.25; bp = [0.25, 0.75]#[-np.inf,-0.2,-0.1,0,0.1,0.2,np.inf]

plotdelta_map(delta_c4,'rmse_vod',cmap="RdBu_r",vmin=-dd,vmax=dd,bp=bp)#,drawmask=False)

plotdelta_map(delta_c4,'rmse_et',cmap='RdBu_r',vmin=-dd,vmax=dd,bp=bp)#,drawmask=False)


#%%
# # plotmap(degrad_pft,varname,cmap='Reds_r',vmin=-2*dd,vmax=0,title="R2(pft)-R2")
# plotmap(delta_c4,varname,cmap='RdBu',vmin=-dd,vmax=dd,title="R2(clusters)-R2(pft)")

plotmap(delta_c4,'rmse_vod',cmap='RdBu_r',vmin=-2,vmax=2,title="RMSE(hft)-RMSE(pft), VOD")
plotmap(delta_c4,'rmse_et',cmap='RdBu_r',vmin=-.5,vmax=.5,title="RMSE(hft)-RMSE(pft), ET")
plotmap(delta_c4,'rmse_sm',cmap='RdBu_r',vmin=-.5,vmax=.5,title="RMSE(hft)-RMSE(pft), SM")
#%%
plotmap(delta_c4,'r2_vod',cmap='RdBu',vmin=-.05,vmax=.05,title="R2(hft)-R2(pft), VOD")
plotmap(delta_c4,'r2_et',cmap='RdBu',vmin=-.05,vmax=.05,title="R2(hft)-R2(pft), ET")
plotmap(delta_c4,'r2_sm',cmap='RdBu',vmin=-.05,vmax=.05,title="R2(hft)-R2(pft), SM")

#%%
# degrad_c40 = degrad_c4.copy()
plt.figure(figsize=(4,6))
vv = 'r2_sm'
a  = degrad_c40[vv].dropna().values
plt.plot(np.sort(a),1-np.arange(len(a))/len(a),label='All')
a  = degrad_c4[vv].dropna().values
plt.plot(np.sort(a),1-np.arange(len(a))/len(a),label='Dry')
plt.legend()
plt.xlabel('Degraded '+vv)
plt.ylabel('edf')
# plotmap(delta_c4-degrad_c40,'r2_vod',cmap='RdBu',vmin=-.01,vmax=.01,title="R2(hft)-R2(pft), VOD")

# plt.xlim(-1,0.02)
# plt.plot([-1,0],[0.5,0.5],'--k')
# a  = degrad_pft['rmse_vod'].dropna().values
# plt.plot(np.sort(a),1-np.arange(len(a))/len(a))
# plt.xlim(-0,1,10)

#%%
dd = 0.15; varname = 'rmse_et'

# count,bins = np.histogram(delta_c4[varname],bins=[-np.inf,-1,-0.1,0,0.1,1,np.inf])
count,bins = np.histogram(delta_c4[varname],bins=[-np.inf,-0.2,-0.1,0,0.1,0.2,np.inf])

cc = [sns.color_palette("RdBu_r",5)[i] for i in [0,1,3,4]]
plt.figure(figsize=(6,3))
for i,itm in enumerate([0,1,4,5]):
    plt.bar(i,count[itm]/len(delta_c4),color=cc[i])
# plt.xticks(np.arange(4),['<-1.0','-1.0~-0.1','0.1~1.0','>1.0'])
plt.xticks(np.arange(4),['<-0.2','-0.2~-0.1','0.1~0.2','>0.2'])

plt.xlabel(r'$\Delta RMSE$')
plt.ylabel('fraction')
plt.title(varname)
# plt.ylim([0,0.5])



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




