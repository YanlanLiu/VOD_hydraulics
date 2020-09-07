#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 10:33:36 2020

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
Collection_STD = np.zeros([len(SiteInfo),11])+np.nan

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
            Collection_STD[subrange,:] = PARA2_std*2
    fname = statspath+'OBS_'+MODE+'_'+str(arrayid).zfill(3)+'.pkl'
    if os.path.isfile(fname):
        with open(fname,'rb') as f:
            OBS_mean,OBS_std,OBS_N = pickle.load(f)            
        if OBS_mean.shape[1]>0:
            Collection_OBS[subrange,:] = OBS_mean
            Collection_N[subrange,:] = OBS_N
            
            

#%%
mycmap = sns.cubehelix_palette(rot=-.63, as_cmap=True)
def plotmap(df,varname,vmin=0,vmax=1,cmap=mycmap,cbartitle=''):
    heatmap1_data = pd.pivot_table(df, values=varname, index='row', columns='col')
    heatmap1_data[578] = np.nan
    plt.figure(figsize=(13.2,5))
    m = Basemap(llcrnrlon = -180.0, llcrnrlat = -60.0, urcrnrlon = 180.0, urcrnrlat = 90.0)
    m.drawcoastlines()
    m.drawcountries()
    lat,lon = LatLon(np.array(heatmap1_data.index),np.array(list(heatmap1_data)))
    cs = m.pcolormesh(lon,lat,heatmap1_data,cmap=cmap,vmin=vmin,vmax=vmax,shading='flat')
    cbar = m.colorbar(cs)
    cbar.set_label(cbartitle,rotation=360,labelpad=23)
    plt.show()
    return 0
    


#%%from matplotlib.colors import LinearSegmentedColormap
# cm = LinearSegmentedColormap.from_list('cm0', sns.color_palette("hls", 15), N=15)
# plotmap(SiteInfo,'IGBP',vmin=0,vmax=15,cmap=cm)

#%%
df_acc = pd.DataFrame(np.concatenate([Collection_ACC,Collection_N],axis=1),columns=['r2_vod','r2_et','r2_sm','Geweke','N_vod','N_et','N_sm'])
df_acc['row'] = SiteInfo['row'];df_acc['col'] = SiteInfo['col']
# plotmap(df_acc,'r2_sm',cbartitle=r'$R^2_{SM}$')
df_acc['r2_sm'][df_acc['N_sm']<df_acc['N_sm'].quantile(.40)] = np.nan
plotmap(df_acc,'r2_sm',cbartitle=r'$R^2_{SM}$')
plotmap(df_acc,'r2_et',cbartitle=r'$R^2_{ET}$')
plotmap(df_acc,'r2_vod',cbartitle=r'$R^2_{VOD}$')


#%%
sns.kdeplot(df_acc['r2_vod'])
sns.kdeplot(df_acc['r2_et'])
sns.kdeplot(df_acc['r2_sm'])
plt.xlabel('R2')
plt.ylabel('pdf')
#%%
df_std = pd.DataFrame(Collection_STD,columns=varnames)
df_std['row'] = SiteInfo['row'];df_std['col'] = SiteInfo['col']; df_std['IGBP'] = SiteInfo['IGBP']
plotmap(df_std,'psi50X',vmin=0,vmax=5,cmap='RdYlBu_r',cbartitle=r'$\psi_{50,X}$')
plotmap(df_std,'g1',vmin=0,vmax=2,cmap='RdYlBu_r',cbartitle=r'$g_1$')



#%%
df_para = pd.DataFrame(Collection_PARA,columns=varnames+['a','b','c'])
df_para['row'] = SiteInfo['row'];df_para['col'] = SiteInfo['col']; df_para['IGBP'] = SiteInfo['IGBP']
df_para['psi50X'] = -df_para['psi50X']
df_para['lpx'] = df_para['lpx']*df_para['psi50X']
#%%
cmap0 = 'RdYlBu'
plotmap(df_para,'psi50X',vmin=-6.5,vmax=-0.1,cmap='RdYlBu',cbartitle=r'$\psi_{50,X}$')
plotmap(df_para,'g1',vmin=0,vmax=6.5,cmap=cmap0,cbartitle=r'$g_1$')
plotmap(df_para,'C',vmin=0,vmax=26,cmap=cmap0,cbartitle=r'$C$')
plotmap(df_para,'lpx',vmin=-3.4,vmax=-0.1,cmap='RdYlBu',cbartitle=r'$\psi_{50,S}$')
plotmap(df_para,'gpmax',vmin=0,vmax=10,cmap=cmap0,cbartitle=r'$g_{p,max}$')
plotmap(df_para,'bc',vmin=0,vmax=.8,cmap=cmap0,cbartitle=r'$bc$')
plotmap(df_para,'bexp',vmin=0,vmax=10,cmap=cmap0,cbartitle=r'$bexp$')

#%%
# plt.plot(df_para['psi50X'],df_para['g1'],'ok')
from Utilities import nancorr
for v in varnames+['a','b','c']:
    print(v,nancorr(df_para['psi50X'].values,df_para[v].values))



#%% PFT-based comparison
IGBP_try = ['GRA','DBF','EBF','SHB','ENF']

TRY = pd.read_excel('../TRY/TRY_Hydraulic_Traits.xlsx')
TRY_P50 = TRY['Water potential at 50% loss of conductivity Psi_50 (MPa)']
TRY_PFT = TRY['PFT']
TRY = pd.DataFrame(np.column_stack([TRY_P50,TRY_PFT]),columns=['P50','PFT'])
TRY = TRY[TRY['P50']!=-999].reset_index()

TRY_P50_mean =[np.median(TRY['P50'][TRY['PFT']==itm]) for itm in IGBP_try]
TRY_P50_min = [np.percentile(TRY['P50'][TRY['PFT']==itm],25) for itm in IGBP_try]
TRY_P50_max = [np.percentile(TRY['P50'][TRY['PFT']==itm],75) for itm in IGBP_try]

IGBPlist = ['NA','ENF','EBF','DNF','DBF','MF','SHB','SHB',
            'SAV','SAV','GRA','WET','CRO','URB','GRA','SNW','NA','NA','NA']

IGBPnames = np.array([IGBPlist[itm] for itm in df_para['IGBP'].values])
IGBPunique = np.unique(IGBPnames)

# List_to_compare = ['Cropland','Grassland','DBF','DNF','Shrubland','ENF']
P50mean = [np.nanmedian(df_para['psi50X'][IGBPnames==itm]) for itm in IGBP_try]
P50_low = [np.nanpercentile(df_para['psi50X'][IGBPnames==itm],25) for itm in IGBP_try]
P50_high = [np.nanpercentile(df_para['psi50X'][IGBPnames==itm],75) for itm in IGBP_try]

#%%
c1 = sns.color_palette("Paired")[0]
c2 = sns.color_palette("Paired")[1]
plt.figure(figsize=(8,8))
dd = 0.3
plt.subplot(211)
for i in range(len(IGBP_try)):
    if i==0:
        plt.bar(i-dd/2,P50mean[i],color=c1,width=dd,label='Estimated')
        plt.plot([i-dd/2,i-dd/2],[P50_low[i],P50_high[i]],'-k')
        plt.bar(i+dd/2,TRY_P50_mean[i],color=c2,width=dd,label=r'TRY')
        
    else:
        plt.bar(i-dd/2,P50mean[i],color=c1,width=dd)
        plt.plot([i-dd/2,i-dd/2],[P50_low[i],P50_high[i]],'-k')
        plt.bar(i+dd/2,TRY_P50_mean[i],color=c2,width=dd)
    plt.plot([i-dd/2,i-dd/2],[P50_low[i],P50_high[i]],'-k')
    plt.plot([i+dd/2,i+dd/2],[TRY_P50_min[i],TRY_P50_max[i]],'-k')
    
plt.xticks([])
plt.ylabel('$\psi_{50,x}$ (MPa)')
plt.xticks(np.arange(len(IGBP_try)),IGBP_try)
plt.legend(bbox_to_anchor=(1.05,1.05))

# c1 = sns.color_palette("Paired")[2]
# c2 = sns.color_palette("Paired")[3]
c1 = sns.color_palette("BrBG_r", 5)[1]
c2 = sns.color_palette("BrBG_r", 5)[0]

IGBP_lin = ['ENF','DBF','SHB','GRA','CRO']
Lin_mean = np.array([2.35,3.97,4.22,4.5,5.79])
Lin_std = np.array([0.25,0.06,0.72,0.37,0.64])
G1mean = [np.nanmean(df_para['g1'][IGBPnames==itm]) for itm in IGBP_lin]
G1_low = [np.nanpercentile(df_para['g1'][IGBPnames==itm],25) for itm in IGBP_lin]
G1_high = [np.nanpercentile(df_para['g1'][IGBPnames==itm],75) for itm in IGBP_lin]

plt.subplot(212)
for i in range(len(IGBP_lin)):
    if i==0:
        plt.bar(i-dd/2,G1mean[i],color=c1,width=dd,label='Estimated')
        plt.bar(i+dd/2,Lin_mean[i],color=c2,width=dd,label=r'Lin et al.')
    else:
        plt.bar(i-dd/2,G1mean[i],color=c1,width=dd)
        plt.bar(i+dd/2,Lin_mean[i],color=c2,width=dd)
    plt.plot([i-dd/2,i-dd/2],[G1_low[i],G1_high[i]],'-k')
    plt.plot([i+dd/2,i+dd/2],Lin_mean[i]+Lin_std[i]*np.array([-1,1]),'-k')
    
    
plt.xticks(np.arange(len(IGBP_lin)),IGBP_lin,rotation=30)
plt.ylim([0,7])
plt.ylabel(r'$g_1$')
plt.legend(bbox_to_anchor=(1.05,1.05))

#%%
import netCDF4
from scipy import ndimage

fp='../CONUS/Trugman_map/CWM_P50_10Deg.nc'
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
lat0,lon0 = LatLon(df_para['row'],df_para['col'])
df_para['lat0'] = lat0; df_para['lon0'] = lon0

new_df = pd.merge(df_para,fia,how='left',left_on=['lat0','lon0'],right_on=['Lat','Lon'])
new_df['IGBPnames'] = IGBPnames
# new_df['psi50X'] = new_df['psi50X']
# new_df = new_df[(IGBPnames!='NA') & (IGBPnames!='Urban') &  (IGBPnames!='Grassland') &  (IGBPnames!='Cropland') ]
if new_df['psi50X'].mean()>0: new_df['psi50X'] = -new_df['psi50X']
plt.figure(figsize=(6,6))
xlim = [-13.5,0.5]
# sns.scatterplot(x="P50", y="psi50X",s=np.log(nplots+1)*100,alpha=0.5, hue="IGBPnames",data=new_df)
sns.scatterplot(x="P50", y="psi50X",s=new_df['nplots']*10,alpha=0.5, hue="IGBPnames",data=new_df)
plt.legend(bbox_to_anchor=(1.75,1.05))
plt.plot(xlim,xlim,'-k');plt.xlim(xlim);plt.ylim(xlim)


