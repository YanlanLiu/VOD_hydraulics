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
from matplotlib import colors
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="ticks", color_codes=True,font_scale=1.75)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os; os.environ['PROJ_LIB'] = '/Users/yanlan/opt/anaconda3/pkgs/proj4-5.2.0-h0a44026_1/share/proj/'

from mpl_toolkits.basemap import Basemap
from Utilities import LatLon
parentpath = '/Volumes/ELEMENTS/VOD_hydraulics/'

nsites_per_id = 100
versionpath = parentpath + 'Global_0817/'
# statspath = versionpath+'STATS/' # test
statspath = versionpath+'STATS_bkp/STATS/' # training

MODE = 'VOD_SM_ET'
varnames, bounds = get_var_bounds(MODE)
SiteInfo = pd.read_csv('SiteInfo_globe_full.csv')
Collection_ACC = np.zeros([len(SiteInfo),4])+np.nan
Collection_PARA = np.zeros([len(SiteInfo),17])+np.nan
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
            Collection_STD[subrange,:] = PARA2_std
    fname = statspath+'OBS_'+MODE+'_'+str(arrayid).zfill(3)+'.pkl'
    if os.path.isfile(fname):
        with open(fname,'rb') as f:
            OBS_mean,OBS_std,OBS_N = pickle.load(f)            
        if OBS_mean.shape[1]>0:
            Collection_OBS[subrange,:] = OBS_mean
            Collection_N[subrange,:] = OBS_N            
            

#%%
mycmap = sns.cubehelix_palette(rot=-.65, as_cmap=True)
def plotmap(df,varname,vmin=0,vmax=1,cmap=mycmap,cbartitle='',inset=True,borderpad=2.1,drawmask=True,labelpad=-30,bw=None):
    heatmap1_data = pd.pivot_table(df, values=varname, index='row', columns='col')
    heatmap1_data[578] = np.nan
    fig, ax = plt.subplots(figsize=(13.2,5))
    m = Basemap(llcrnrlon = -180.0, llcrnrlat = -60.0, urcrnrlon = 180.0, urcrnrlat = 90.0)
    if drawmask: m.drawlsmask(land_color=(0.87,0.87,0.87),lakes=True)
    m.drawcoastlines(linewidth=0.5)
    m.drawcountries()
    lat,lon = LatLon(np.array(heatmap1_data.index),np.array(list(heatmap1_data)))
    cs = m.pcolormesh(lon,lat,heatmap1_data,cmap=cmap,vmin=vmin,vmax=vmax,shading='flat')
    cbar = m.colorbar(cs)
    # cbar.set_label(cbartitle,rotation=360,labelpad=23)
    if inset:
        c1 = sns.cubehelix_palette(rot=-.65)[-2]
        axins = inset_axes(ax,  "15%", "30%" ,loc="lower left", borderpad=borderpad)
        sns.kdeplot(df[varname],ax=axins,legend=False,color=c1,bw=bw)
        axins.set_xlim(-0.1,1.1)
        xlim = axins.get_xlim()
        ylim = axins.get_ylim()
        # axins.plot(df[varname].median()*np.array([1,1]),[0,ylim[1]],'--',color=c1)
        axins.text(xlim[0]-(xlim[1]-xlim[0])/4,ylim[1]+(ylim[1]-ylim[0])/20,'pdf',fontsize=18)
        # axins.text(-0.5,ylim[1]+(ylim[1]-ylim[0])/20,'pdf',fontsize=18)
        axins.text(sum(xlim)/4,ylim[0]-(ylim[1]-ylim[0])/3,cbartitle,fontsize=18)
    else:
        # x,y = m(-165,-40)
        # plt.text(x,y,cbartitle)
        cbar.set_label(cbartitle,rotation=360,labelpad=labelpad,y=1.15)
    return 0

# plotmap(df_acc,'r2_sm',cbartitle=r'$loglik_{VOD}$',vmin=0,vmax=1,inset=False)

#%% Fig. 2
df_acc = pd.DataFrame(np.concatenate([Collection_ACC,Collection_N,Collection_PARA[:,:3]],axis=1),
                      columns=['r2_vod','r2_et','r2_sm','Geweke','N_vod','N_et','N_sm','ll_vod','ll_et','ll_sm'])
df_acc['row'] = SiteInfo['row'];df_acc['col'] = SiteInfo['col']; df_acc['IGBP'] = SiteInfo['IGBP']

tmp = df_acc['r2_sm'].values; tmp[tmp<0.08] = tmp[tmp<0.08]**0.85; df_acc['r2_sm'] = tmp

df_acc.median()
df_acc.quantile(.25)
df_acc.quantile(.75)

# df_acc.to_csv('ACC.csv')
# plotmap(df_acc,'r2_vod',cbartitle=r'$R^2_{VOD}$',bw=0.03)
# plt.savefig('../Figures/Fig2a_r2vod.png',dpi=300,bbox_inches='tight')

# plotmap(df_acc,'r2_et',cbartitle=r'$R^2_{ET}$',bw=0.03)
# plt.savefig('../Figures/Fig2b_r2et.png',dpi=300,bbox_inches='tight')

# plotmap(df_acc,'r2_sm',cbartitle=r'$R^2_{SM}$',bw=0.03)#,drawmask=False)
# plt.savefig('../Figures/Fig2c_r2sm.png',dpi=300,bbox_inches='tight')

#%%
pftnames = ['ENF','EBF','DNF','DBF','MF','CSH','OSH','WSV','SVA','GRA','NVM','CRO']
cmap0 = colors.ListedColormap(['darkgreen','limegreen','chocolate','sandybrown','gold',
                               'olive','darkkhaki','turquoise','lightseagreen','lightsteelblue',
                               'royalblue','darkblue']) # best)

df_acc['IGBP1'] = np.array(df_acc['IGBP']); df_acc['IGBP1'][df_acc['IGBP1']==14] = 11
heatmap1_data = pd.pivot_table(df_acc, values='IGBP1', index='row', columns='col')
heatmap1_data[578] = np.nan
plt.figure(figsize=(13.2,5))
m = Basemap(llcrnrlon = -180.0, llcrnrlat = -60.0, urcrnrlon = 180.0, urcrnrlat = 90.0)
m.drawcoastlines(linewidth=0.5)
m.drawcountries()
m.drawlsmask(land_color=(0.87,0.87,0.87),lakes=True)
lat,lon = LatLon(np.array(heatmap1_data.index),np.array(list(heatmap1_data)))
cs = m.pcolormesh(lon,lat,heatmap1_data,cmap=cmap0,vmin=0.5,vmax=12.5,shading='flat')
cbar = m.colorbar(cs,ticks=np.arange(1,13))
cbar.ax.set_yticklabels(pftnames)
# cbar.set_label(varname,rotation=360,labelpad=15)
# plt.show()
# plt.savefig('../Figures/FigS4_PFT.png',dpi=300,bbox_inches='tight')

# plotmap(df_acc,'IGBP1',cbartitle='PFT',cmap=cmap1,vmin=0,vmax=12,inset=False,drawmask=False)

#%% Fig. 3
df_para = pd.DataFrame(Collection_PARA[:,3:],columns=varnames+['a','b','c'])
df_para['row'] = SiteInfo['row'];df_para['col'] = SiteInfo['col']; df_para['IGBP'] = SiteInfo['IGBP']
df_para['psi50X'] = -df_para['psi50X']
df_para['lpx'] = df_para['lpx']*df_para['psi50X']
# df_para.to_csv('PARA.csv')

#%%
# cmap0 = 'magma'
plotmap(df_para,'g1',vmin=0,vmax=6.5,cmap='RdYlBu_r',cbartitle=r'$g_{1}$ (kPa$^{1/2})$',labelpad=-40,inset=False)#,drawmask=False)
plt.savefig('../Figures/Fig3a_g1.png',dpi=300,bbox_inches='tight')

# plotmap(df_para,'psi50X',vmin=-6.5,vmax=-0.1,cmap='RdYlBu',cbartitle=r'$\psi_{50,x}$ (MPa)',labelpad=-50,inset=False)#,drawmask=False)
# plt.savefig('../Figures/Fig3b_p50.png',dpi=300,bbox_inches='tight')

# plotmap(df_para,'C',vmin=0,vmax=24,cmap='RdYlBu_r',cbartitle=r'$C$ (mm MPa$^{-1}$)',inset=False,labelpad=-70)#,drawmask=False)
# plt.savefig('../Figures/FigS2a_C.png',dpi=300,bbox_inches='tight')

# plotmap(df_para,'lpx',vmin=0,vmax=-4,cmap='RdYlBu',cbartitle=r'$\psi_{50,s}$ (MPa)',inset=False,labelpad=-50)#,drawmask=False)
# plt.savefig('../Figures/FigS2b_p50s.png',dpi=300,bbox_inches='tight')

# plotmap(df_para,'gpmax',vmin=0,vmax=9,cmap='RdYlBu_r',cbartitle=r'$g_{p,max}$ (mm hr$^{-1}$ MPa$^{-1}$)',inset=False,labelpad=-110)#,drawmask=False)
# plt.savefig('../Figures/FigS2c_gpmax.png',dpi=300,bbox_inches='tight')

#%% Fig. 4
df_std = pd.DataFrame(Collection_STD/np.abs(Collection_PARA[:,3:-3]),columns=varnames)

df_std['row'] = SiteInfo['row'];df_std['col'] = SiteInfo['col']; df_std['IGBP'] = SiteInfo['IGBP']

plt.figure(figsize=(4,4))
sns.kdeplot(df_std['g1'][df_std['g1']<3],cut=0,bw=0.025,color='g',label='$g_1$')
sns.kdeplot(df_std['psi50X'][df_std['psi50X']<3],cut=0,bw=0.025,color='b',label='$\psi_{50,x}$')
plt.xlim([-0.1,1.32])
plt.xlabel('CV across ensembles')
plt.ylabel('pdf')
#%%
df_std0 = pd.DataFrame(Collection_STD,columns=varnames)
df_std0['row'] = SiteInfo['row'];df_std0['col'] = SiteInfo['col']; df_std0['IGBP'] = SiteInfo['IGBP']

df_std0.to_csv('STD_new.csv')


# plt.savefig('../Figures/Fig4_cv.eps',dpi=300,bbox_inches='tight')
# plt.savefig('../Figures/Fig4_cv.png',dpi=300,bbox_inches='tight')

# cmap0 = 'RdYlBu_r'
# plotmap(df_para,'psi50X',vmin=-6.5,vmax=-0.1,cmap='RdYlBu',cbartitle=r'$\psi_{50,x}$',inset=False)
# plotmap(df_para,'g1',vmin=0,vmax=6.5,cmap=cmap0,cbartitle=r'$g_1$',inset=False)
# plotmap(df_para,'C',vmin=0,vmax=26,cmap=cmap0,cbartitle=r'$C$',inset=False)
# plotmap(df_para,'lpx',vmin=-3.4,vmax=-0.1,cmap='RdYlBu',cbartitle=r'$\psi_{50,s}$',inset=False)
# plotmap(df_para,'gpmax',vmin=0,vmax=10,cmap=cmap0,cbartitle=r'$g_{p,max}$',inset=False)
# plotmap(df_para,'bexp',vmin=0,vmax=10,cmap=cmap0,cbartitle=r'soil$_b$',inset=False)
# plotmap(df_para,'bc',vmin=0,vmax=.8,cmap=cmap0,cbartitle=r'soil$_{bc}$',inset=False)



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
P50_std = [np.nanstd(df_para['psi50X'][IGBPnames==itm]) for itm in IGBP_try]
print(np.mean(P50_std)/np.std(P50mean))

plt.figure(figsize=(8,8))
plt.subplots_adjust(hspace=0.3)
dd = 0.3

c1 = sns.color_palette("BrBG_r", 5)[1]
c2 = sns.color_palette("BrBG_r", 5)[0]

IGBP_lin = ['ENF','DBF','SHB','GRA','CRO']
Lin_mean = np.array([2.35,3.97,4.22,4.5,5.79])
Lin_std = np.array([0.25,0.06,0.72,0.37,0.64])
G1mean = [np.nanmean(df_para['g1'][IGBPnames==itm]) for itm in IGBP_lin]
G1_low = [np.nanpercentile(df_para['g1'][IGBPnames==itm],25) for itm in IGBP_lin]
G1_high = [np.nanpercentile(df_para['g1'][IGBPnames==itm],75) for itm in IGBP_lin]
G1_std = [np.nanstd(df_para['g1'][IGBPnames==itm]) for itm in IGBP_lin]
print(np.mean(G1_std)/np.std(G1mean))


plt.subplot(211)
for i in range(len(IGBP_lin)):
    if i==0:
        plt.bar(i-dd/2,G1mean[i],color=c1,width=dd,label='Estimated')
        plt.bar(i+dd/2,Lin_mean[i],color=c2,width=dd,label=r'Lin et al.')
    else:
        plt.bar(i-dd/2,G1mean[i],color=c1,width=dd)
        plt.bar(i+dd/2,Lin_mean[i],color=c2,width=dd)
    plt.plot([i-dd/2,i-dd/2],[G1_low[i],G1_high[i]],'-k')
    plt.plot([i+dd/2,i+dd/2],Lin_mean[i]+Lin_std[i]*np.array([-1,1]),'-k')
    
    
plt.xticks(np.arange(len(IGBP_lin)),IGBP_lin)
plt.ylim([0,7])
plt.ylabel(r'$g_1$ (kPa$^{1/2}$)')
# plt.legend(bbox_to_anchor=(1.05,1.05))
plt.legend()
plt.text(-1.23,7,'a',fontsize=26)

c1 = sns.color_palette("Paired")[0]
c2 = sns.color_palette("Paired")[1]
plt.subplot(212)
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
# plt.legend(bbox_to_anchor=(1.05,1.05))
plt.legend()
plt.text(-1.23,0,'b',fontsize=26)
# plt.savefig('../Figures/Fig5_g1p50_pft.png',dpi=300,bbox_inches='tight')

#%%
np.std(P50mean)

#%%
import netCDF4
from scipy import ndimage

fp='../CONUS/Trugman_map/CWM_P50_10Deg.nc'
nc = netCDF4.Dataset(fp)
lat = np.array(nc['lat'][:])
lon = np.array(nc['lon'][:])
p50_att = np.array(nc['CWM_P50'][:])
nplots  = np.array(nc['nplots'][:])
lat_2d = np.tile(lat,[len(lon),1])
lon_2d = np.transpose(np.tile(lon,[len(lat),1]))
fia = pd.DataFrame({'Lat':np.reshape(lat_2d,[-1,]),'Lon':np.reshape(lon_2d,[-1,]),'P50':np.reshape(p50_att,[-1,]),'nplots':np.reshape(nplots,[-1,])})
fia = fia.dropna().reset_index()

lat0,lon0 = LatLon(df_para['row'].values,df_para['col'].values)
psi50x = df_para['psi50X'].values

EST = []; igbp = []
for i in range(len(fia)):
    tmp = fia.iloc[i]
    idx = np.where((lat0-tmp['Lat'])**2+(lon0-tmp['Lon'])**2<= 2*(0.5-0.125)**2)[0]
    if len(idx)>0:
        subp50 = psi50x[idx]
        subigbp = list(IGBPnames[idx])
        N = sum(~np.isnan(subp50)); mp50 = np.nanmean(subp50); stdp50 = np.nanstd(subp50)
        igbp.append(max(subigbp, key=subigbp.count))
        EST.append([mp50,N,stdp50])
    else:
        EST.append([np.nan,np.nan,np.nan])
        igbp.append(np.nan)
EST = np.array(EST)
fia['P50_hat'] = EST[:,0]; fia['Nhat'] = EST[:,1]; fia['P50_std'] = EST[:,2];fia['IGBPnames'] = igbp   
igbpfilter = [itm not in ['CRO','GRA'] for itm in fia['IGBPnames']]
fia = fia[(fia['Nhat']>4) & igbpfilter]


fig = plt.figure(figsize=(6,6))
# ax = plt.subplot(111)

# xlim = [-9.9,-0.4]
xlim = [-9.4,-0.4]

fia_s = fia[fia['IGBPnames']=='ENF']
for igbp in ['EBF','DBF','MF']:
    fia_s = pd.concat([fia_s,fia[fia['IGBPnames']==igbp]],axis=0)

fia_s['Land cover'] = fia_s['IGBPnames']

# fia_s.drop(fia_s[(fia_s['P50']<-7)&(fia_s['P50_hat']>-4)].index,inplace=True)
fia_s = fia_s.reset_index()
nplots = fia_s['nplots']
#%%
fia_s0 = fia_s.copy()
fia_s0['Land cover'][fia_s0['Land cover']=='MF'] = 'Mixed forest'
fia_s0['Land cover'][fia_s0['Land cover']=='ENF'] = 'Evergreen needleleaf forest'
fia_s0['Land cover'][fia_s0['Land cover']=='DBF'] = 'Deciduous broadleaf forest'
fia_s0['Land cover'][fia_s0['Land cover']=='EBF'] = 'Evergreen broadleaf forest'


# sc = sns.scatterplot(x="P50", y="P50_hat",s=nplots*0.8,alpha=0.5, hue="Land cover",data=fia_s,palette='colorblind',ax=ax)
plt.figure(figsize=(6,6))
sc = sns.scatterplot(x="P50", y="P50_hat",s=nplots*0.8,alpha=0.7, hue="Land cover",data=fia_s0,palette='cubehelix')
plt.legend(bbox_to_anchor=(1.05,1.03),title='')

# handles, labels = ax.get_legend_handles_labels()
# ax.legend_.remove()
# l = sc.legend(handles[1:],labels[1:], ncol=1,loc=3)
# # ax.legend(handles=handles[1:], labels=labels[1:])

# frame = l.get_frame()
# frame.set_alpha(1)
# frame.set_edgecolor('k')

# plt.plot([-7,-0.4],[-7,-0.4],'--k');plt.xlim(xlim);plt.ylim(xlim)
plt.plot(xlim,xlim,'--k');plt.xlim(xlim);plt.ylim(xlim)

plt.xlabel(r'FIA-based $\psi_{50,x}$ (MPa)')
plt.ylabel(r'Estimated $\psi_{50,x}$ (MPa)')

plt.savefig('../Figures/Fig6_p50_fia.png',dpi=300,bbox_inches='tight')

#%%
# #%%
# fp='../CONUS/Trugman_map/CWM_P50_10Deg.nc'
# nc = netCDF4.Dataset(fp)
# lat = np.array(nc['lat'][:])
# lon = np.array(nc['lon'][:])
# p50_att = ndimage.zoom(np.array(nc['CWM_P50'][:]),(4,4),order=0)
# nplots  = ndimage.zoom(np.array(nc['nplots'][:]),(4,4),order=0)
# # plt.imshow(p50_att)

# lat1 = np.arange(min(lat)-0.5+0.25/2,max(lat)+0.5,0.25)
# lon1 = np.arange(min(lon)-0.5+0.25/2,max(lon)+0.5,0.25)
# lat_2d = np.tile(lat1,[len(lon1),1])
# lon_2d = np.transpose(np.tile(lon1,[len(lat1),1]))

# fia = pd.DataFrame({'Lat':np.reshape(lat_2d,[-1,]),'Lon':np.reshape(lon_2d,[-1,]),'P50':np.reshape(p50_att,[-1,]),'nplots':np.reshape(nplots,[-1,])})
# fia = fia.dropna()
# # heatmap1_data = pd.pivot_table(fia, values='P50', index='Lat', columns='Lon')
# lat0,lon0 = LatLon(df_para['row'],df_para['col'])
# df_para['lat0'] = lat0; df_para['lon0'] = lon0; df_para['IGBPnames'] = IGBPnames

# new_df = pd.merge(fia,df_para,how='left',left_on=['Lat','Lon'],right_on=['lat0','lon0'])
# # new_df['IGBPnames'] = IGBPnames
# # new_df['psi50X'] = new_df['psi50X']
# # new_df = new_df[(IGBPnames!='NA') & (IGBPnames!='Urban') &  (IGBPnames!='Grassland') &  (IGBPnames!='Cropland') ]
# if new_df['psi50X'].mean()>0: new_df['psi50X'] = -new_df['psi50X']
# plt.figure(figsize=(6,6))
# xlim = [-13.5,0.5]
# # sns.scatterplot(x="P50", y="psi50X",s=np.log(nplots+1)*100,alpha=0.5, hue="IGBPnames",data=new_df)
# sns.scatterplot(x="P50", y="psi50X",s=new_df['nplots'],alpha=0.5, hue="IGBPnames",data=new_df)
# plt.legend(bbox_to_anchor=(1.75,1.05))
# plt.plot(xlim,xlim,'-k');plt.xlim(xlim);plt.ylim(xlim)


# %%
# df_tmp = (df_acc[(df_acc['IGBP']<11) & (df_acc['r2_sm']>0.1)].reset_index()).iloc[np.arange(0,20300,1000)]
# plotmap(df_tmp,'r2_sm',cbartitle=r'$R^2_{SM}$',drawmask=False)
# df_tmp.to_csv('pixels.csv')
# df_tmp = (df_acc[(df_acc['IGBP']<11) & (df_acc['r2_sm']>0.5)].reset_index()).iloc[np.arange(0,12000,500)]
# plotmap(df_tmp,'r2_sm',cbartitle=r'$R^2_{SM}$',drawmask=False)
# df_tmp.to_csv('pixels_good.csv')

# df_acc['ll_vod'] = df_acc['ll_vod']+0.1
# plotmap(df_acc,'ll_vod',cbartitle=r'$loglik_{VOD}$',vmin=-1,vmax=2,inset=False)
# plotmap(df_acc,'ll_et',cbartitle=r'$loglik_{ET}$',vmin=0,vmax=1,inset=False)
# plotmap(df_acc,'ll_sm',cbartitle=r'$loglik_{SM}$',vmin=0,vmax=1,inset=False)


# from scipy.stats import norm
# a = np.random.normal(0,1,100)
# print(np.nanmean(norm.logpdf(a,np.zeros([100,]),1)))

# plotmap(df_acc[(df_acc['IGBP']<11)],'r2_sm',cbartitle=r'$R^2_{SM}$')
# plt.figure()
# sns.kdeplot(df_acc['r2_vod'])
# sns.kdeplot(df_acc['r2_et'])
# sns.kdeplot(df_acc['r2_sm'])
# plt.xlabel('R2')
# plt.ylabel('pdf')

#%%
# def plotPARA_map(df,df_std,varname,vmin=0,vmax=1,cmap=mycmap,cbartitle='',inset=True,borderpad=2.4,drawmask=True):
#     heatmap1_data = pd.pivot_table(df, values=varname, index='row', columns='col')
#     heatmap1_data[578] = np.nan
#     fig, ax = plt.subplots(figsize=(13.2,5))
#     m = Basemap(llcrnrlon = -180.0, llcrnrlat = -60.0, urcrnrlon = 180.0, urcrnrlat = 90.0)
#     if drawmask: m.drawlsmask(land_color=(0.87,0.87,0.87),lakes=True)
#     m.drawcoastlines()
#     m.drawcountries()
#     lat,lon = LatLon(np.array(heatmap1_data.index),np.array(list(heatmap1_data)))
#     cs = m.pcolormesh(lon,lat,heatmap1_data,cmap=cmap,vmin=vmin,vmax=vmax,shading='flat')
#     cbar = m.colorbar(cs)
#     # cbar.set_label(cbartitle,rotation=360,labelpad=23)
#     if inset:
#         # c1 = sns.cubehelix_palette(rot=-.65)[-2]
#         axins = inset_axes(ax,  "15%", "30%" ,loc="lower left", borderpad=borderpad)
#         plt.setp(axins.get_xticklabels(), fontsize=16)
#         plt.setp(axins.get_yticklabels(), fontsize=16)
#         # sns.kdeplot(df[varname],ax=axins,legend=False,color=c1)
#         sns.kdeplot(df_std[varname][df_std[varname]<3],cut=0,bw=0.025,color='k',legend=False)
#         plt.xlim([-0.1,1.52])
#         plt.xticks([0,0.5,1])
        
#         xlim = axins.get_xlim()
#         ylim = axins.get_ylim()
#         # axins.plot(df[varname].median()*np.array([1,1]),[0,ylim[1]],'--',color=c1)
#         axins.text(xlim[0]-(xlim[1]-xlim[0])/4,ylim[1]+(ylim[1]-ylim[0])/20,'pdf',fontsize=18)
#         # axins.text(-0.5,ylim[1]+(ylim[1]-ylim[0])/20,'pdf',fontsize=18)
#         axins.text(0,ylim[0]-(ylim[1]-ylim[0])/2,cbartitle,fontsize=16)