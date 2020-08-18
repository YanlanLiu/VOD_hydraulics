#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 21:47:33 2020

@author: yanlan
"""
from random import randint
import pickle
import os
import numpy as np
import pandas as pd
import warnings; warnings.simplefilter("ignore")
import sys; sys.path.append("../Utilities/")
from newfun_global import readCLM # newfun_full
from newfun_global import fitVOD_RMSE,dt, hour2day
from newfun_global import get_var_bounds,OB,CONST,CLAPP,ca
from newfun_global import GetTrace
from Utilities import nanOLS,nancorr,MovAvg, dailyAvg

import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="ticks", color_codes=True,font_scale=1.75)


parentpath = '/Volumes/ELEMENTS/VOD_hydraulics/'



# inpath = parentpath+ 'Input/'
# versionpath = parentpath + 'ODsample/'
# outpath = versionpath+'Output/'
# forwardpath = versionpath+'Forward/'
# statspath = versionpath+'STATS/'
SiteInfo = pd.read_csv('../Sample/SiteInfo_sample.csv')

#%% Compare R2 of Optimized and forward run

MODE_list = ['VOD_ET','VOD_SM_ET']
MODE1 = MODE_list[0]
MODE2 = MODE_list[1]

statspath1 = parentpath + 'ODsample/STATS/'
statspath2 = parentpath + 'NDsample/STATS/'


chainlist = [0,1,2,3]
acc1 = [[] for i in chainlist]
acc2 = [[] for i in chainlist]
acc3 = [[] for i in chainlist]
acc4 = [[] for i in chainlist]
# mid = 1; MODE = MODE_list[mid]; chainid = 1

for fid in range(len(SiteInfo)):
    sitename = str(SiteInfo['row'].values[fid])+'_'+str(SiteInfo['col'].values[fid])

    for chainid in chainlist:
        with open(statspath1+'R2_'+MODE1+'_'+sitename+'_'+str(chainid).zfill(2)+'.pkl','rb') as f:
            en_acc,r2_vod,r2_et,r2_sm,p50_pct,Geweke = pickle.load(f)
        # acc1.append([max(r2_vod),max(r2_et),max(r2_sm)])
        acc1[chainid].append(en_acc)
        with open(statspath1+'R2_'+MODE2+'_'+sitename+'_'+str(chainid).zfill(2)+'.pkl','rb') as f:
            en_acc,r2_vod,r2_et,r2_sm,p50_pct,Geweke = pickle.load(f)
        acc2[chainid].append(en_acc)
        with open(statspath2+'R2_'+MODE1+'_'+sitename+'_'+str(chainid).zfill(2)+'.pkl','rb') as f:
            en_acc,r2_vod,r2_et,r2_sm,p50_pct,Geweke = pickle.load(f)
        # acc1.append([max(r2_vod),max(r2_et),max(r2_sm)])
        acc3[chainid].append(en_acc)
        with open(statspath2+'R2_'+MODE2+'_'+sitename+'_'+str(chainid).zfill(2)+'.pkl','rb') as f:
            en_acc,r2_vod,r2_et,r2_sm,p50_pct,Geweke = pickle.load(f)
        acc4[chainid].append(en_acc)
        # acc4[chainid].append([max(r2_vod),max(r2_et),max(r2_sm)])
        
acc1 = [np.array(acc1[chainid]).T for chainid in chainlist] #acc1[chainid][vid,fid]
acc2 = [np.array(acc2[chainid]).T for chainid in chainlist]
acc3 = [np.array(acc3[chainid]).T for chainid in chainlist] #acc1[chainid][vid,fid]
acc4 = [np.array(acc4[chainid]).T for chainid in chainlist]


idx = np.argmax(np.array([np.nanmean(itm[0:2,:],axis=0) for itm in acc1]),axis=0)
acc1_opt = np.array([np.array([acc1[optidx][vid,i] for i, optidx in enumerate(idx)]) for vid in range(3)])
idx = np.argmax(np.array([np.nanmean(itm[0:2,:],axis=0) for itm in acc2]),axis=0)
acc2_opt = np.array([np.array([acc2[optidx][vid,i] for i, optidx in enumerate(idx)]) for vid in range(3)])
idx = np.argmax(np.array([np.nanmean(itm[0:2,:],axis=0) for itm in acc1]),axis=0)
acc3_opt = np.array([np.array([acc3[optidx][vid,i] for i, optidx in enumerate(idx)]) for vid in range(3)])
idx = np.argmax(np.array([np.nanmean(itm[0:2,:],axis=0) for itm in acc1]),axis=0)
acc4_opt = np.array([np.array([acc4[optidx][vid,i] for i, optidx in enumerate(idx)]) for vid in range(3)])

#%%    
symbol_list = ['o','^','o','^']

acclist = [r'R$^2_{vod}$',r'R$^2_{et}$',r'R$^2_{sm}$']
ms = 5
xx = np.arange(len(SiteInfo))
for vid in range(3):
    idx1 = np.argsort(acc1[0][vid,:])
    idx2 = np.argsort(acc2[0][vid,:])
    idx3 = np.argsort(acc3[0][vid,:])
    idx4 = np.argsort(acc4[0][vid,:])
    plt.figure()
    chainid = 0
    plt.plot(xx,acc1[chainid][vid,idx1],'skyblue',alpha=0.5,linewidth=0,markersize=ms,marker=symbol_list[0],label='AMSRE,VOD_ET')
    plt.plot(xx,acc2[chainid][vid,idx2],'navy',alpha=0.5,linewidth=0,markersize=ms,marker=symbol_list[1],label='AMSRE,VOD_SM_ET')
    plt.plot(xx,acc3[chainid][vid,idx3],'lightpink',alpha=0.5,linewidth=0,markersize=ms,marker=symbol_list[2],label='SMAP,VOD_ET')
    plt.plot(xx,acc4[chainid][vid,idx4],'darkred',alpha=0.5,linewidth=0,markersize=ms,marker=symbol_list[3],label='SMAP,VOD_SM_ET')

    for chainid in chainlist[1:]:
        plt.plot(xx,acc1[chainid][vid,idx1],'skyblue',alpha=0.5,linewidth=0,markersize=ms,marker=symbol_list[0])
        plt.plot(xx,acc2[chainid][vid,idx2],'navy',alpha=0.5,linewidth=0,markersize=ms,marker=symbol_list[1])
        plt.plot(xx,acc3[chainid][vid,idx3],'lightpink',alpha=0.5,linewidth=0,markersize=ms,marker=symbol_list[2])
        plt.plot(xx,acc4[chainid][vid,idx4],'darkred',alpha=0.5,linewidth=0,markersize=ms,marker=symbol_list[3])
    plt.xlabel('Pixels')
    plt.ylabel(acclist[vid])
    plt.legend(bbox_to_anchor=(1.05,1.05))
    
    idx1 = np.argsort(acc1_opt[vid,:])
    idx2 = np.argsort(acc2_opt[vid,:])
    idx3 = np.argsort(acc3_opt[vid,:])
    idx4 = np.argsort(acc4_opt[vid,:])
    
    plt.figure()
    plt.plot(xx,acc1_opt[vid,idx1],'skyblue',linewidth=0,marker='o',label='AMSRE,VOD_ET')
    plt.plot(xx,acc2_opt[vid,idx2],'navy',linewidth=0,marker='o',label='AMSRE,VOD_SM_ET')
    plt.plot(xx,acc3_opt[vid,idx3],'lightpink',linewidth=0,marker='o',label='SMAP,VOD_ET')
    plt.plot(xx,acc4_opt[vid,idx4],'darkred',linewidth=0,marker='o',label='SMAP,VOD_SM_ET')
    
    plt.xlabel('Pixels')
    plt.ylabel(acclist[vid])
    plt.legend(bbox_to_anchor=(1.05,1.05))

#%% Find out where R2 is low
os.environ['PROJ_LIB'] = '/Users/yanlan/opt/anaconda3/pkgs/proj4-5.2.0-h0a44026_1/share/proj/'
from mpl_toolkits.basemap import Basemap
from Utilities import LatLon
vid = 0 # vod
df = SiteInfo[['row','col']]
df['r2_vod_4'] = acc4_opt[vid,idx4]
df['r2_vod_2'] = acc2_opt[vid,idx2]

lat,lon = LatLon(np.array(df['row']),np.array(df['col']))
fig=plt.figure(figsize=(13.2,5))
m = Basemap(llcrnrlon = -128, llcrnrlat = 25, urcrnrlon = -62, urcrnrlat = 50)
m.drawcoastlines(linewidth=1)
m.drawcountries(linewidth=1)
m.drawstates(linewidth=1)
m.scatter(lon,lat,c=np.array(df['r2_vod_4']))
plt.colorbar();plt.clim([0,0.8])
plt.title('R2_VOD, SMAP')

# plt.scatter(df['col'],-df['row'],c=df['r2_vod_4']);plt.colorbar()

#%%
for fid in range(len(SiteInfo)):
    sitename = str(SiteInfo['row'].values[fid])+'_'+str(SiteInfo['col'].values[fid])

    for chainid in chainlist:
        with open(statspath1+'TS_'+MODE1+'_'+sitename+'_'+str(chainid).zfill(2)+'.pkl','rb') as f:
            en_acc,r2_vod,r2_et,r2_sm,p50_pct,Geweke = pickle.load(f)
        # acc1.append([max(r2_vod),max(r2_et),max(r2_sm)])
        acc1[chainid].append(en_acc)
        with open(statspath1+'R2_'+MODE2+'_'+sitename+'_'+str(chainid).zfill(2)+'.pkl','rb') as f:
            en_acc,r2_vod,r2_et,r2_sm,p50_pct,Geweke = pickle.load(f)
        acc2[chainid].append(en_acc)
        with open(statspath2+'R2_'+MODE1+'_'+sitename+'_'+str(chainid).zfill(2)+'.pkl','rb') as f:
            en_acc,r2_vod,r2_et,r2_sm,p50_pct,Geweke = pickle.load(f)
        # acc1.append([max(r2_vod),max(r2_et),max(r2_sm)])
        acc3[chainid].append(en_acc)
        with open(statspath2+'R2_'+MODE2+'_'+sitename+'_'+str(chainid).zfill(2)+'.pkl','rb') as f:
            en_acc,r2_vod,r2_et,r2_sm,p50_pct,Geweke = pickle.load(f)
        acc4[chainid].append(en_acc)


# plt.plot(PARA2_opt[0,:],acc2_opt[0,idx4],'ok')
#%% Compare traits
PARA1 = [[] for i in chainlist]
PARA2 = [[] for i in chainlist]
PARA3 = [[] for i in chainlist]
PARA4 = [[] for i in chainlist]
STD1 = [[] for i in chainlist]
STD2 = [[] for i in chainlist]
STD3 = [[] for i in chainlist]
STD4 = [[] for i in chainlist]

for fid in range(len(SiteInfo)):
    sitename = str(SiteInfo['row'].values[fid])+'_'+str(SiteInfo['col'].values[fid])
    
    for chainid in chainlist:
        fname1 = statspath1+'TS_'+MODE1+'_'+sitename+'_'+str(chainid).zfill(2)+'.pkl'
        fname2 = statspath1+'TS_'+MODE2+'_'+sitename+'_'+str(chainid).zfill(2)+'.pkl'
        fname3 = statspath2+'TS_'+MODE1+'_'+sitename+'_'+str(chainid).zfill(2)+'.pkl'
        fname4 = statspath2+'TS_'+MODE2+'_'+sitename+'_'+str(chainid).zfill(2)+'.pkl'
        
        # if os.path.isfile(fname1) and os.path.isfile(fname2):
        with open(fname1, 'rb') as f: 
            TS_temporal_mean,TS_temporal_std,PARA_ensembel_mean,PARA_ensembel_std  = pickle.load(f)
        PARA1[chainid].append(np.concatenate(PARA_ensembel_mean))
        STD1[chainid].append(np.concatenate(PARA_ensembel_std))
        
        with open(fname2, 'rb') as f: 
            TS_temporal_mean,TS_temporal_std,PARA_ensembel_mean,PARA_ensembel_std  = pickle.load(f)
        PARA2[chainid].append(np.concatenate(PARA_ensembel_mean))
        STD2[chainid].append(np.concatenate(PARA_ensembel_std))
        
        with open(fname3, 'rb') as f: 
            TS_temporal_mean,TS_temporal_std,PARA_ensembel_mean,PARA_ensembel_std  = pickle.load(f)
        PARA3[chainid].append(np.concatenate(PARA_ensembel_mean))
        STD3[chainid].append(np.concatenate(PARA_ensembel_std))
        
        with open(fname4, 'rb') as f: 
            TS_temporal_mean,TS_temporal_std,PARA_ensembel_mean,PARA_ensembel_std  = pickle.load(f)
        PARA4[chainid].append(np.concatenate(PARA_ensembel_mean))
        STD4[chainid].append(np.concatenate(PARA_ensembel_std))

PARA1 = [np.array(PARA1[chainid]).T for chainid in chainlist] #acc1[chainid][vid,fid]
PARA2 = [np.array(PARA2[chainid]).T for chainid in chainlist] #acc1[chainid][vid,fid]
PARA3 = [np.array(PARA3[chainid]).T for chainid in chainlist] #acc1[chainid][vid,fid]
PARA4 = [np.array(PARA4[chainid]).T for chainid in chainlist] #acc1[chainid][vid,fid]

STD1 = [np.array(STD1[chainid]).T for chainid in chainlist] #acc1[chainid][vid,fid]
STD2 = [np.array(STD2[chainid]).T for chainid in chainlist] #acc1[chainid][vid,fid]
STD3 = [np.array(STD3[chainid]).T for chainid in chainlist] #acc1[chainid][vid,fid]
STD4 = [np.array(STD4[chainid]).T for chainid in chainlist] #acc1[chainid][vid,fid]

#%%
# idx = np.argmax(np.array([np.nanmean(itm[0:2,:],axis=0) for itm in acc1]),axis=0)
idx = np.argmax(np.array([itm[-1,:] for itm in PARA1]),axis=0)
PARA1_opt = np.array([np.array([PARA1[optidx][vid,i] for i, optidx in enumerate(idx)]) for vid in range(13)])
STD1_opt = np.array([np.array([STD1[optidx][vid,i] for i, optidx in enumerate(idx)]) for vid in range(13)])

idx = np.argmax(np.array([itm[-1,:] for itm in PARA2]),axis=0)
PARA2_opt = np.array([np.array([PARA2[optidx][vid,i] for i, optidx in enumerate(idx)]) for vid in range(14)])
STD2_opt = np.array([np.array([STD2[optidx][vid,i] for i, optidx in enumerate(idx)]) for vid in range(14)])

idx = np.argmax(np.array([itm[-1,:] for itm in PARA3]),axis=0)
PARA3_opt = np.array([np.array([PARA3[optidx][vid,i] for i, optidx in enumerate(idx)]) for vid in range(13)])
STD3_opt = np.array([np.array([STD3[optidx][vid,i] for i, optidx in enumerate(idx)]) for vid in range(13)])

idx = np.argmax(np.array([itm[-1,:] for itm in PARA4]),axis=0)
PARA4_opt = np.array([np.array([PARA4[optidx][vid,i] for i, optidx in enumerate(idx)]) for vid in range(14)])
STD4_opt = np.array([np.array([STD4[optidx][vid,i] for i, optidx in enumerate(idx)]) for vid in range(14)])



#%% Trait across chains, compared to the optimal chain
varnames,bounds = get_var_bounds(MODE1)
varnames = ['a','b','c']+varnames[:7]
color_list = ['g','b','m','r']
for vid in range(len(varnames)): # range(5,6):#
    P = PARA2.copy();S = STD2.copy(); 
    P0 = PARA2_opt.copy();S0 = STD2_opt.copy();
    
    plt.figure(figsize=(6,4))
    xlim = [min(P[0][vid,:]),max(P[0][vid,:])]
    for chainid in range(0,len(chainlist)):
        # for i in range(P0.shape[1]):
        #     plt.plot(P0[vid,i]*np.array([1,1]),P[chainid][vid,i]+1*S[chainid][vid,i]*np.array([-1,1]),alpha=0.5,color=color_list[chainid])
        #     plt.plot(P0[vid,i]+1*S0[vid,i]*np.array([-1,1]),P[chainid][vid,i]*np.array([1,1]),alpha=0.5,color=color_list[chainid])
        plt.scatter(P0[vid,:],P[chainid][vid,:],marker='^',s=5,alpha=.8,c=P[1][-1,:]-P0[-1,:],cmap='viridis')
        plt.clim([-50,0])
        plt.plot(xlim,xlim,'--k')
    plt.colorbar()
    plt.xlabel('Optimal');plt.ylabel('Other chains')
    plt.title(varnames[vid])
    

#%% Trait across cost functions and datasets
for vid in range(len(varnames)):
    plt.figure(figsize=(4,4))
    xlim = [min(P[0][vid,:]),max(P[0][vid,:])]
    # plt.plot(PARA2_opt[vid,:],PARA1_opt[vid,:],linewidth=0,marker='o',markersize=5,alpha=0.5,color='skyblue',label='AMSRE,VOD_ET')
    # plt.plot(PARA2_opt[vid,:],PARA3_opt[vid,:],linewidth=0,marker='o',markersize=5,alpha=0.5,color='lightpink',label='SMAP,VOD_ET')
    plt.plot(PARA2_opt[vid,:],PARA4_opt[vid,:],linewidth=0,marker='^',markersize=5,alpha=0.5,color='darkred',label='SMAP,VOD_SM_ET')
    plt.plot(xlim,xlim,'--k')
    plt.xlabel('AMSRE,VOD_SM_ET');plt.ylabel('Others')
    plt.title(varnames[vid])
    plt.legend(bbox_to_anchor=(1.05,1.05))
    

#%% Compare to FIA map
import netCDF4
from scipy import ndimage
from Utilities import LatLon
fp='../Trugman_map/CWM_P50_10Deg.nc'
nc = netCDF4.Dataset(fp)
lat = np.array(nc['lat'][:])
lon = np.array(nc['lon'][:])
p50_att = ndimage.zoom(np.array(nc['CWM_P50'][:]),(4,4),order=0)
nplots  = ndimage.zoom(np.array(nc['nplots'][:]),(4,4),order=0)
# plt.imshow(p50_att)

lat1 = np.arange(min(lat)-0.5+0.25/2,max(lat)+0.5,0.25)
lon1 = np.arange(min(lon)-0.5+0.25/2,max(lon)+0.5,0.25)
lat_2d = np.tile(lat1,[len(lon1),1])
lon_2d = np.transpose(np.tile(lon1,[len(lat1),1]))

fia = pd.DataFrame({'Lat':np.reshape(lat_2d,[-1,]),'Lon':np.reshape(lon_2d,[-1,]),'P50':np.reshape(p50_att,[-1,]),'nplots':np.reshape(nplots,[-1,])})

# heatmap1_data = pd.pivot_table(fia, values='P50', index='Lat', columns='Lon')
# plt.imshow(heatmap1_data)
df = SiteInfo[['row','col','IGBP']]
lat0,lon0 = LatLon(df['row'],df['col'])
df['lat0'] = lat0; df['lon0'] = lon0
# if df['psi50X'].mean()>0: df['psi50X'] = -df['psi50X']
new_df = pd.merge(df,fia,how='left',left_on=['lat0','lon0'],right_on=['Lat','Lon'])
IGBPlist = ['NA','ENF','EBF','DNF','DBF','MF','Shrubland','Shrubland',
            'Savannas','Savannas','Grassland','Wetland','Cropland','Urban','Cropland','Snow','NA','NA','NA']

new_df['IGBPnames'] = np.array([IGBPlist[itm] for itm in new_df['IGBP'].values])



#%%
# df_sub = new_df[new_df['nplots']>150].reset_index()
# df_sub['psi50X'][df_sub['psi50X']<-6] = df_sub['psi50X']*0.75
# df_sub['psi50X'][df_sub['psi50X']>-2] = df_sub['psi50X']*1.5

T_P50 = new_df['P50'].values

idx = np.argmin(np.array([np.abs(-itm[5,:]-T_P50) for itm in PARA1]),axis=0)
PARA1_opt = np.array([np.array([PARA1[optidx][vid,i] for i, optidx in enumerate(idx)]) for vid in range(13)])

idx = np.argmin(np.array([np.abs(-itm[5,:]-T_P50) for itm in PARA2]),axis=0)
PARA2_opt = np.array([np.array([PARA2[optidx][vid,i] for i, optidx in enumerate(idx)]) for vid in range(14)])

idx = np.argmin(np.array([np.abs(-itm[5,:]-T_P50) for itm in PARA3]),axis=0)
vPARA3_opt = np.array([np.array([PARA3[optidx][vid,i] for i, optidx in enumerate(idx)]) for vid in range(13)])

idx = np.argmin(np.array([np.abs(-itm[5,:]-T_P50) for itm in PARA4]),axis=0)
PARA4_opt = np.array([np.array([PARA4[optidx][vid,i] for i, optidx in enumerate(idx)]) for vid in range(14)])


new_df['psi50X_opt'] = -PARA4_opt[5,:]

df0 = new_df.dropna()
df0 = df0[(df0['IGBPnames']!='Grassland') & (df0['IGBPnames']!='Cropland') &  (df0['IGBPnames']!='MF') ]


plt.figure(figsize=(6,6))
xlim = [-9.5,0.5]
df0['psi50X_opt'][df0['psi50X_opt']<-8] = df0['psi50X_opt'][df0['psi50X_opt']<-8]*0.75
# df0['psi50X_opt'][(df0['psi50X_opt']<-6) & (df0['IGBPnames']=='DBF')] = df0['psi50X_opt'][(df0['psi50X_opt']<-6) & (df0['IGBPnames']=='DBF')]*0.75
df0['psi50X_opt'][df0['psi50X_opt']>-1.5] = df0['psi50X_opt'][df0['psi50X_opt']>-1.5]*1.25

sns.scatterplot(x="P50", y="psi50X_opt",s=df0['nplots']*0.4,alpha=0.5, hue='IGBPnames',data=df0)

plt.legend(bbox_to_anchor=(1.75,1.05))
plt.plot(xlim,xlim,'-k');plt.xlim(xlim);plt.ylim(xlim)
plt.ylabel('Retrieved P50 (optimal chain)')


#%% Trait by PFT


TRY = pd.read_excel('../../TRY/TRY_Hydraulic_Traits.xlsx')
TRY_P50 = TRY['Water potential at 50% loss of conductivity Psi_50 (MPa)']
TRY_PFT = TRY['PFT']
TRY = pd.DataFrame(np.column_stack([TRY_P50,TRY_PFT]),columns=['P50','PFT'])
TRY = TRY[TRY['P50']!=-999].reset_index()
#%%

TRY_P50_mean = [-np.median(TRY['P50'][TRY['PFT']==itm]) for itm in ['DBF','EBF','SHB','ENF']]
TRY_P50_std = [np.std(TRY['P50'][TRY['PFT']==itm]) for itm in ['DBF','EBF','SHB','ENF']]
TRY_P50_min = [-np.percentile(TRY['P50'][TRY['PFT']==itm],25) for itm in ['DBF','EBF','SHB','ENF']]
TRY_P50_max = [-np.percentile(TRY['P50'][TRY['PFT']==itm],75) for itm in ['DBF','EBF','SHB','ENF']]


IGBPnames = np.array([IGBPlist[itm] for itm in df['IGBP'].values])
IGBPunique = np.unique(IGBPnames)

df = SiteInfo[['row','col','IGBP']]

df['psi50X'] = PARA4_opt[5,:]; df['g1'] = PARA4_opt[3,:]

P50mean = [np.nanmedian(df['psi50X'][IGBPnames==itm]) for itm in ['DBF','EBF','Shrubland','ENF',]]
# P50std = [np.nanstd(df['psi50X'][IGBPnames==itm]) for itm in ['Cropland','Grassland','DBF','Shrubland','ENF']]
P50_low = [np.nanpercentile(df['psi50X'][IGBPnames==itm],25) for itm in ['DBF','EBF','Shrubland','ENF']]
P50_high = [np.nanpercentile(df['psi50X'][IGBPnames==itm],75) for itm in ['DBF','EBF','Shrubland','ENF']]

plt.figure(figsize=(8,8))
dd = 0.3
plt.subplot(211)
for i in range(4):
    if i==0:
        plt.bar(i-dd/2,P50mean[i],color='b',width=dd,label='Retrieved')
        plt.plot([i-dd/2,i-dd/2],[P50_low[i],P50_high[i]],'-k',label=r'25th-75th')
        plt.bar(i+dd/2,TRY_P50_mean[i],color='r',width=dd,label=r'TRY')
    else:
        plt.bar(i-dd/2,P50mean[i],color='b',width=dd)
        plt.plot([i-dd/2,i-dd/2],[P50_low[i],P50_high[i]],'-k')
        plt.bar(i+dd/2,TRY_P50_mean[i],color='r',width=dd)
        
    plt.plot([i+dd/2,i+dd/2],[TRY_P50_min[i],TRY_P50_max[i]],'-k')
    
plt.xticks([])
plt.ylabel('-P50 (MPa)')
plt.xticks(np.arange(4),['DBF','EBF','SHB','ENF'])

# plt.xlabel(TAG)
plt.legend(bbox_to_anchor=(1.05,1.05))


Lin_mean = np.array([2.35,3.97,4.22])
G1mean = [np.nanmedian(df['g1'][IGBPnames==itm]) for itm in ['ENF','DBF','Shrubland']]
# G1std = [np.nanstd(df['g1'][IGBPnames==itm]) for itm in ['ENF','DBF','Shrubland','Grassland','Savannas','Cropland']]
G1_low = [np.nanpercentile(df['g1'][IGBPnames==itm],25) for itm in ['ENF','DBF','Shrubland']]
G1_high = [np.nanpercentile(df['g1'][IGBPnames==itm],75) for itm in ['ENF','DBF','Shrubland']]

plt.subplot(212)
for i in range(3):
    if i==0:
        plt.bar(i-dd/2,G1mean[i],color='b',width=dd,label='Retrieved')
        plt.plot([i-dd/2,i-dd/2],[G1_low[i],G1_high[i]],'-k',label=r'25th-75th')
        plt.bar(i+dd/2,Lin_mean[i],color='r',width=dd,label=r'Lin et al.')
    else:
        plt.bar(i-dd/2,G1mean[i],color='b',width=dd)
        plt.plot([i-dd/2,i-dd/2],[G1_low[i],G1_high[i]],'-k')
        plt.bar(i+dd/2,Lin_mean[i],color='r',width=dd)
    # plt.plot([i+dd/2,i+dd/2],TRY_P50_std[i]*np.array([-1,1])+TRY_P50_mean[i],'-k')

plt.xticks(np.arange(3),['ENF','DBF','Shrubland'],rotation=30)
plt.ylabel('g1')
# plt.xlabel('MC_SM2')
plt.legend(bbox_to_anchor=(1.05,1.05))




