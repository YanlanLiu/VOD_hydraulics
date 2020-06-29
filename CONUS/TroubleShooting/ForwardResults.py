#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 23:29:34 2020

@author: yanlan

Stats of forward runs
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns; sns.set(style="ticks", color_codes=True,font_scale=1.5)

parentpath = '/Volumes/ELEMENTS/VOD_hydraulics/'
TAG = 'SMslope'#'Control'
versionpath = parentpath + 'TroubleShooting/'+TAG+'/'
statspath = versionpath+'STATS/'
obspath = versionpath+'../OBS_STATS/'
MODE = 'AM_PM_ET'
SiteInfo = pd.read_csv('SiteInfo_reps.csv')

Nsample = 100
varlist = ['VOD','S1','ET',r'VOD$_{p/a}$',r'VOD$_{w/d}$',r'ET$_{w/d}$','ISO']
p = len(varlist)
BIAS = np.zeros([p,Nsample])
CVE = np.zeros([p,Nsample])
mLAI =  np.zeros([Nsample,])

acclist =[r'R$^2_{VOD}$',r'R$^2_{ET}$',r'R$^2_{SM}$',r'P50$_{flt}$']
ACC = np.zeros([len(acclist),Nsample])
ACC1 = np.zeros([len(acclist),Nsample])
for fid in set(range(Nsample)):#-set([12,13]):
    print(fid)
    sitename = str(SiteInfo['row'][fid])+'_'+str(SiteInfo['col'][fid])
    with open(obspath+MODE+'_'+sitename+'.pkl','rb') as f:
        OBS_temporal_mean,OBS_temporal_std = pickle.load(f)
    with open(statspath+MODE+'_'+sitename+'.pkl' , 'rb') as f: 
        TS_temporal_mean,TS_temporal_std,PARA_ensembel_mean,PARA_ensembel_std = pickle.load(f)    
    with open(statspath+'R2_'+sitename+'.pkl', 'rb') as f: 
        r2_vod,r2_et,r2_sm,p50_pct = pickle.load(f)
    TS_temporal_mean[4][TS_temporal_mean[4]<-15] = np.nan
    TSmm = np.array([np.mean(itm[np.isfinite(itm)]) for itm in TS_temporal_mean])
    TSmstd = np.array([np.abs(np.nanmean(TS_temporal_std[i]/TS_temporal_mean[i])) for i in range(len(TS_temporal_mean))])
    TSmstde = np.abs(np.array([np.std(itm[np.isfinite(itm)]) for itm in TS_temporal_mean])) # std across ensembles (temporal mean)
    
    
    OBSm = np.array(OBS_temporal_mean[:3]+OBS_temporal_mean[4:])
    OBScv = np.array(OBS_temporal_std[:3]+OBS_temporal_std[:4])/OBSm
    mLAI[fid] = np.copy(OBS_temporal_mean[3])
    TSm = np.array([TSmm[0],TSmm[5],TSmm[1]+TSmm[2],TSmm[7],TSmm[9],TSmm[10],TSmm[11]])
    TScvt = np.array([TSmstd[0],TSmstd[5],TSmstd[1]+TSmstd[2],TSmstd[7],TSmstd[9],TSmstd[10],TSmstd[11]])/OBSm
    TScve = np.array([TSmstde[0],TSmstde[5],TSmstde[1]+TSmstde[2],TSmstde[7],TSmstde[9],TSmstde[10],TSmstde[11]])/OBSm
    # OBSm=OBS_temporal_mean[:3]+OBS_temporal_mean[4:]
    # VOD,SOILM,ET,dLAI,VODr_ampm,VODr_wd,ETr_wd,ISO = OBS_temporal_mean or OBS_temporal_std
    # # VOD,E,T,ET_AP,PSIL,S1,S2,VODr_ampm, ETr_ampm, VODr_wd, ETr_wd, ISO= TS_temporal_mean or TS_temporal_std
    # # popt, theta = PARA_ensembel_mean,PARA_ensembel_std 

    
    # plt.plot(TSm/OBSm-1,'ob')
    # plt.plot([-.2,len(OBSm)-.8],[0,0],'-',color='grey')
    # delta = 0.1
    # for i in range(len(OBSm)):
    #     plt.plot([i-delta,i-delta],OBScv[i]*np.array([-1,1]),'-k')
    #     plt.plot([i,i],TScvt[i]*np.array([-1,1])+TSm[i]/OBSm[i]-1,'-b')
    #     plt.plot([i+delta,i+delta],TScve[i]*np.array([-1,1])+TSm[i]/OBSm[i]-1,'-r')
    # plt.xticks(np.arange(len(OBSm)),varlist)
    # plt.ylabel('Diff. to Obs.')
    BIAS[:,fid] = TSm/OBSm-1
    CVE[:,fid] = TScve
    ACC[:,fid] = np.array([max(r2_vod),max(r2_et),max(r2_sm),(p50_pct[2]-p50_pct[0])/7.5])
    
    summary_name = './Summary_'+TAG+'.pkl'
    with open(summary_name,'wb') as f: pickle.dump((BIAS,CVE,ACC,mLAI),f)
       


#%%

plt.figure(figsize=(6,4))
tmpfilter = (mLAI>0.9);print(sum(tmpfilter))
sns.heatmap(BIAS[:,tmpfilter],vmin=-1,vmax=1,cmap='RdBu')
plt.xticks([])
plt.yticks(np.arange(7)+0.5,varlist,rotation=0)
plt.title('Bias')
plt.xlabel('Pixels')
  
plt.figure(figsize=(6,4))
sns.heatmap(CVE[:,tmpfilter],vmin=0,vmax=1,cmap='Blues')
plt.xticks([])
plt.yticks(np.arange(p)+0.5,varlist,rotation=0)
plt.title('CV across ensembles')  
plt.xlabel('Pixels')

plt.figure(figsize=(6,4))
sns.heatmap(ACC[:,tmpfilter],vmin=-1,vmax=1,cmap='RdBu')
plt.xticks([])
plt.yticks(np.arange(4)+0.5,acclist,rotation=0)
plt.title('Accuracy and flatness')  
plt.xlabel('Pixels')


#%%
with open('./Summary_Weights.pkl','rb') as f: 
    BIAS0,CVE0,ACC0,mLAI = pickle.load(f)

with open('./Summary_SMslope.pkl','rb') as f: 
    BIAS1,CVE1,ACC1,mLAI = pickle.load(f)
    
tmpfilter = (mLAI>0.9)

tmp = ACC1[:,tmpfilter]-ACC0[:,tmpfilter]
improved_fraction = 1-np.sum(tmp<0,axis=1)/tmp.shape[1]; improved_fraction[-1] = 1-improved_fraction[-1]
print(improved_fraction)


plt.figure(figsize=(6,4))
sns.heatmap(ACC1[:,tmpfilter]-ACC0[:,tmpfilter],vmin=-1,vmax=1,cmap='RdBu')
plt.xticks([])
plt.yticks(np.arange(4)+0.5,acclist,rotation=0)
plt.title('SM-CTL')  
plt.xlabel('Pixels')




#%%
# VOD,SOILM,ET,VODr_ampm,VODr_wd,ETr_wd,ISO
# with open(obsname, 'wb') as f: 
#    pickle.dump((OBS_temporal_mean,OBS_temporal_std), f)
# parentpath = '/Volumes/ELEMENTS/VOD_hydraulics/'
#arrayid = 81
#nsites_per_id = 1
#warmup, nsample,thinning = (0.8,2,10)