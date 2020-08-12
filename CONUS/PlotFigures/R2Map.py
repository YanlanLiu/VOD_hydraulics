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
df['psi50X'] = PARA_all[:,2]
df['g1'] = PARA_all[:,0]

df['row'] = SiteInfo['row'].iloc[:len(df)]
df['col'] = SiteInfo['col'].iloc[:len(df)]
df['IGBP'] = SiteInfo['IGBP'].iloc[:len(df)]
df = df.dropna().reset_index()

lat,lon = LatLon(np.array(df['row']),np.array(df['col']))


#%%
varname = 'R2_SM'; vmin = 0; vmax = 1

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

#%%
# df = pd.DataFrame(ACC_all,columns=['R2_VOD','R2_ET','R2_SM','Geweke'])
# df = pd.concat([df,SiteInfo.iloc[:len(df)]],axis=1)

TRY = pd.read_excel('../../TRY/TRY_Hydraulic_Traits.xlsx')
TRY_P50 = TRY['Water potential at 50% loss of conductivity Psi_50 (MPa)']
TRY_PFT = TRY['PFT']
TRY = pd.DataFrame(np.column_stack([TRY_P50,TRY_PFT]),columns=['P50','PFT'])
TRY = TRY[TRY['P50']!=-999].reset_index()
#%%
TRY_P50_mean = [-np.median(TRY['P50'][TRY['PFT']==itm]) for itm in ['GRA','DBF','SHB','ENF']]
TRY_P50_std = [np.std(TRY['P50'][TRY['PFT']==itm]) for itm in ['GRA','DBF','SHB','ENF','GRA']]
TRY_P50_min = [-np.percentile(TRY['P50'][TRY['PFT']==itm],25) for itm in ['GRA','DBF','SHB','ENF']]
TRY_P50_max = [-np.percentile(TRY['P50'][TRY['PFT']==itm],75) for itm in ['GRA','DBF','SHB','ENF']]

IGBPlist = ['NA','ENF','EBF','DNF','DBF','MF','Shrubland','Shrubland',
            'Savannas','Savannas','Grassland','Wetland','Cropland','Urban','Cropland','Snow','NA','NA','NA']

IGBPnames = np.array([IGBPlist[itm] for itm in df['IGBP'].values])
IGBPunique = np.unique(IGBPnames)

P50mean = [np.nanmean(df['psi50X'][IGBPnames==itm]) for itm in ['Cropland','Grassland','DBF','Shrubland','ENF',]]
# P50std = [np.nanstd(df['psi50X'][IGBPnames==itm]) for itm in ['Cropland','Grassland','DBF','Shrubland','ENF']]
P50_low = [np.nanpercentile(df['psi50X'][IGBPnames==itm],25) for itm in ['Cropland','Grassland','DBF','Shrubland','ENF']]
P50_high = [np.nanpercentile(df['psi50X'][IGBPnames==itm],75) for itm in ['Cropland','Grassland','DBF','Shrubland','ENF']]

#%%
plt.figure(figsize=(8,8))
dd = 0.3
plt.subplot(211)
for i in range(5):
    if i==0:
        plt.bar(i-dd/2,P50mean[i],color='b',width=dd,label='Retrieved')
        plt.plot([i-dd/2,i-dd/2],[P50_low[i],P50_high[i]],'-k',label=r'25th-75th')
        
    else:
        plt.bar(i-dd/2,P50mean[i],color='b',width=dd)
        plt.plot([i-dd/2,i-dd/2],[P50_low[i],P50_high[i]],'-k')
        if i==1:
            plt.bar(i+dd/2,TRY_P50_mean[i-1],color='r',width=dd,label=r'TRY')
        else:
            plt.bar(i+dd/2,TRY_P50_mean[i-1],color='r',width=dd)
        
        plt.plot([i+dd/2,i+dd/2],[TRY_P50_min[i-1],TRY_P50_max[i-1]],'-k')
    
plt.xticks([])
plt.ylabel('-P50 (MPa)')
plt.xticks(np.arange(5),['CRO','GRA','DBF','SHB','ENF'])

# plt.xlabel(TAG)
plt.legend(bbox_to_anchor=(1.05,1.05))


Lin_mean = np.array([2.35,3.97,4.22,4.5,5.76,5.79])
G1mean = [np.nanmean(df['g1'][IGBPnames==itm]) for itm in ['ENF','DBF','Shrubland','Grassland','Savannas','Cropland']]
# G1std = [np.nanstd(df['g1'][IGBPnames==itm]) for itm in ['ENF','DBF','Shrubland','Grassland','Savannas','Cropland']]
G1_low = [np.nanpercentile(df['g1'][IGBPnames==itm],25) for itm in ['ENF','DBF','Shrubland','Grassland','Savannas','Cropland']]
G1_high = [np.nanpercentile(df['g1'][IGBPnames==itm],75) for itm in ['ENF','DBF','Shrubland','Grassland','Savannas','Cropland']]

plt.subplot(212)
for i in range(6):
    if i==0:
        plt.bar(i-dd/2,G1mean[i],color='b',width=dd,label='Retrieved')
        plt.plot([i-dd/2,i-dd/2],[G1_low[i],G1_high[i]],'-k',label=r'25th-75th')
        plt.bar(i+dd/2,Lin_mean[i],color='r',width=dd,label=r'Lin et al.')
    else:
        plt.bar(i-dd/2,G1mean[i],color='b',width=dd)
        plt.plot([i-dd/2,i-dd/2],[G1_low[i],G1_high[i]],'-k')
        plt.bar(i+dd/2,Lin_mean[i],color='r',width=dd)
    # plt.plot([i+dd/2,i+dd/2],TRY_P50_std[i]*np.array([-1,1])+TRY_P50_mean[i],'-k')

plt.xticks(np.arange(6),['ENF','DBF','Shrubland','Grassland','Savannas','Cropland'],rotation=30)
plt.ylabel('g1')
# plt.xlabel('MC_SM2')
plt.legend(bbox_to_anchor=(1.05,1.05))

#%%
df_trees = df[(IGBPnames!='Grassland')]

# varname = 'g1'
varname='psi50X'
# if df[varname].mean()>0:df[varname] = -df[varname]
lat,lon = LatLon(np.array(df_trees['row']),np.array(df_trees['col']))

heatmap1_data = pd.pivot_table(df_trees, values=varname, index='row', columns='col')
fig=plt.figure(figsize=(13.2,5))
m = Basemap(llcrnrlon = -128, llcrnrlat = 25, urcrnrlon = -62, urcrnrlat = 50)

m.drawcoastlines(linewidth=1)
m.drawcountries(linewidth=1)
m.drawstates(linewidth=1)
mycmap = sns.cubehelix_palette(rot=-.63, as_cmap=True)
# mycmap = sns.cubehelix_palette(6, rot=-.5, dark=.3,as_cmap=True)
cs = m.pcolormesh(np.unique(lon),np.flipud(np.unique(lat)),heatmap1_data,cmap=mycmap,vmin=0,vmax=12,shading='quad')
cbar = m.colorbar(cs)
cbar.set_label(varname,rotation=360,labelpad=15)
plt.show()

#%% Compare to Trugman's map
# list(df)
import netCDF4
from scipy import ndimage

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

heatmap1_data = pd.pivot_table(fia, values='P50', index='Lat', columns='Lon')
# plt.imshow(1heatmap1_data)

#%%
IGBPnames = np.array([IGBPlist[itm] for itm in df['IGBP'].values])

lat0,lon0 = LatLon(df['row'],df['col'])
# df0 = df.copy()
df['lat0'] = lat0; df['lon0'] = lon0
# df['psi50X'][df['psi50X']<-6] = df['psi50X']*0.75
# df['psi50X'][df['psi50X']>-2] = df['psi50X']*1.5

# df['psi50X'][(df['psi50X']<-6) & (df['IGBP']!=6) & (df['IGBP']!=7)] = df['psi50X']<-6) & (df['IGBP']!=6) & (df['IGBP']!=7),2]*0.75


if df['psi50X'].mean()>0: df['psi50X'] = -df['psi50X']
new_df = pd.merge(df,fia,how='left',left_on=['lat0','lon0'],right_on=['Lat','Lon'])
new_df['IGBPnames'] = IGBPnames
# new_df['psi50X'] = new_df['psi50X']
# new_df = new_df[(IGBPnames!='NA') & (IGBPnames!='Urban') &  (IGBPnames!='Grassland') &  (IGBPnames!='Cropland') & (df['N_VOD']>10) & (df['N_ET']>2)]
new_df = new_df[(IGBPnames!='NA') & (IGBPnames!='Urban') &(IGBPnames!='Savannas') &(IGBPnames!='MF') & (IGBPnames!='Grassland') &  (IGBPnames!='Cropland')]

plt.figure(figsize=(6,6))
xlim = [-13.5,0.5]
sns.scatterplot(x="P50", y="psi50X",s=new_df['nplots']*0.1,alpha=0.5, hue="IGBPnames",data=new_df)
# sns.scatterplot(x="P50", y="psi50X",s=20,alpha=0.5, hue="IGBPnames",data=new_df)

plt.legend(bbox_to_anchor=(1.75,1.05))
plt.plot(xlim,xlim,'-k');plt.xlim(xlim);plt.ylim(xlim)


#%%
heatmap1_data = pd.pivot_table(new_df, values='nplots', index='Lat', columns='Lon')
plt.figure()
plt.imshow(np.flipud(heatmap1_data),cmap=mycmap);plt.colorbar()
plt.title('# of plots')
plt.figure()
plt.hist(new_df['nplots'][new_df['nplots']>1])

#%%
plt.figure(figsize=(6,6))
xlim = [-13.5,0.5]
df_sub = new_df[new_df['nplots']>150].reset_index()
# df_sub['psi50X'][df_sub['psi50X']<-6] = df_sub['psi50X']*0.75
# df_sub['psi50X'][df_sub['psi50X']>-2] = df_sub['psi50X']*1.5

sns.scatterplot(x="P50", y="psi50X",s=df_sub['nplots']*.4,alpha=0.5, hue="IGBPnames",data=df_sub)
# sns.scatterplot(x="P50", y="psi50X",s=20,alpha=0.5, hue="IGBPnames",data=new_df)

plt.legend(bbox_to_anchor=(1.75,1.05))
plt.plot(xlim,xlim,'-k');plt.xlim(xlim);plt.ylim(xlim)
print(len(df_sub))

df_sub[['row','col']].to_csv('FIA_plots.csv')
#%%
# lat,lon = LatLon(np.array(new_df['row']),np.array(new_df['col']))

# fig=plt.figure(figsize=(13.2,5))
# m = Basemap(llcrnrlon = -128, llcrnrlat = 25, urcrnrlon = -62, urcrnrlat = 50)

# m.drawcoastlines(linewidth=1)
# m.drawcountries(linewidth=1)
# m.drawstates(linewidth=1)
# mycmap = sns.cubehelix_palette(rot=-.63, as_cmap=True)
# # mycmap = sns.cubehelix_palette(6, rot=-.5, dark=.3,as_cmap=True)
# cs = m.pcolormesh(np.unique(lon),np.flipud(np.unique(lat)),heatmap1_data,cmap=mycmap,vmin=0,vmax=1,shading='quad')
# cbar = m.colorbar(cs)
# cbar.set_label(varname,rotation=360,labelpad=15)
# plt.show()

#%%
# Trait = pd.DataFrame(V50,columns=varnames[:V50.shape[1]])

# df = pd.concat([SiteInfo,Trait,VN],axis=1)
# df['Geweke'] = Geweke
# df['psi50X'] = -df['psi50X']


# df['psi50X'][(df['psi50X']<-6) & (df['IGBP']!=6) & (df['IGBP']!=7)] = -V25[(df['psi50X']<-6) & (df['IGBP']!=6) & (df['IGBP']!=7),2]*0.75
# df['psi50X'][(df['psi50X']>-2) & (df['IGBP']!=6) & (df['IGBP']!=7)] = -V75[(df['psi50X']>-2) & (df['IGBP']!=6) & (df['IGBP']!=7),2]


# lat0,lon0 = LatLon(df['row'],df['col'])

# IGBPnames = np.array([IGBPlist[itm] for itm in df['IGBP'].values])

# df['lat0'] = lat0; df['lon0'] = lon0; df['IGBPnames'] = IGBPnames
# df_1deg = df.copy()
# lcfilter = np.array([itm in ['ENF','DBF','MF','Savannas','Shrubland'] for itm in IGBPnames])
# cvgfilter = np.array(df['Geweke']<1)
# obsfilter = np.array((df['N_VOD']>200) & (df['N_ET']>10))# to be used

# df_1deg = df_1deg.iloc[ (obsfilter)]
# new_df = pd.merge(df_1deg,fia,how='left',left_on=['lat0','lon0'],right_on=['Lat','Lon'])
# plt.figure(figsize=(6,6))
# xlim = [-13.5,0.5]
# sns.scatterplot(x="P50", y="psi50X",s=20, hue="IGBPnames",data=new_df,palette="Set1")
# plt.legend(bbox_to_anchor=(1.75,1.05))
# plt.plot(xlim,xlim,'-k');plt.xlim(xlim);plt.ylim(xlim)
# plt.xlabel('P50 from Trugman et al.')
# plt.ylabel('Retrieved P50')

# # plt.figure()
# heatmap1_data = pd.pivot_table(df_1deg, values='psi50X', index='row', columns='col')
# # plt.imshow(heatmap1_data);plt.colorbar()

# lat,lon = LatLon(np.array(df_1deg['row']),np.array(df_1deg['col']))

# fig=plt.figure(figsize=(13.2,5))
# m = Basemap(llcrnrlon = -128, llcrnrlat = 25, urcrnrlon = -62, urcrnrlat = 50)

# m.drawcoastlines(linewidth=1)
# m.drawcountries(linewidth=1)
# m.drawstates(linewidth=1)
# mycmap = sns.cubehelix_palette(rot=-.63, as_cmap=True,reverse=True)
# # mycmap = sns.cubehelix_palette(6, rot=-.5, dark=.3,as_cmap=True)
# cs = m.pcolormesh(np.unique(lon),np.flipud(np.unique(lat)),heatmap1_data,cmap=mycmap,vmin=-7,vmax=0,shading='quad')
# cbar = m.colorbar(cs)
# cbar.set_label(varname,rotation=360,labelpad=15)
# plt.show()

# ENF_DBF = new_df.iloc[[itm in ['ENF','DBF','MF'] for itm in new_df['IGBPnames']]]
# print(nancorr(ENF_DBF['psi50X'],ENF_DBF['P50']))

# ENF_DBF = new_df.iloc[[itm in ['ENF','DBF','MF','Shrubland'] for itm in new_df['IGBPnames']]]
# print(nancorr(SHB['psi50X'],SHB['P50']))


