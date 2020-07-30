#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 22:55:20 2020

@author: yanlan
"""

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
TAG = 'Test'#'Control'
versionpath = parentpath + 'OSSE2/'+TAG+'/'
statspath = versionpath+'STATS/'
MODE = 'VOD_SM_ET'
SiteInfo = pd.read_csv('SiteInfo_reps_53.csv')

Nsample = len(SiteInfo)
varlist = ['VOD','S1','ET',r'VOD$_{p/a}$',r'VOD$_{w/d}$',r'ET$_{w/d}$','ISO']
p = len(varlist)
BIAS = np.zeros([p,Nsample])
CVE = np.zeros([p,Nsample])
mLAI =  np.zeros([Nsample,])

acclist =[r'R$^2_{VOD}$',r'R$^2_{ET}$',r'R$^2_{SM}$',r'P50$_{flt}$','Geweke']
ACC = np.zeros([len(acclist),Nsample])
ACC1 = np.zeros([len(acclist),Nsample])
for fid in set(range(Nsample)):#-set([12,13]):
    print(fid)
    sitename = str(SiteInfo['row'][fid])+'_'+str(SiteInfo['col'][fid])
    with open(statspath+'OBS_'+MODE+'_'+sitename+'.pkl','rb') as f:
        OBS_temporal_mean,OBS_temporal_std = pickle.load(f)
    with open(statspath+'TS_'+MODE+'_'+sitename+'.pkl' , 'rb') as f: 
        TS_temporal_mean,TS_temporal_std,PARA_ensembel_mean,PARA_ensembel_std = pickle.load(f)    
    with open(statspath+'R2_'+MODE+'_'+sitename+'.pkl', 'rb') as f: 
        accen, r2_vod,r2_et,r2_sm,p50_pct, Geweke = pickle.load(f)
    TS_temporal_mean[4][TS_temporal_mean[4]<-15] = np.nan
    TSmm = np.array([np.mean(itm[np.isfinite(itm)]) for itm in TS_temporal_mean])
    TSmstd = np.array([np.abs(np.nanmean(TS_temporal_std[i]/TS_temporal_mean[i])) for i in range(len(TS_temporal_mean))])
    TSmstde = np.abs(np.array([np.std(itm[np.isfinite(itm)]) for itm in TS_temporal_mean])) # std across ensembles (temporal mean)
    
    
    # OBSm = np.array(OBS_temporal_mean[:3]+OBS_temporal_mean[4:])
    # OBScv = np.array(OBS_temporal_std[:3]+OBS_temporal_std[4:])/OBSm
    OBSm = np.array(OBS_temporal_mean)
    OBScv = np.array(OBS_temporal_std)/OBSm
    mLAI[fid] = np.copy(OBS_temporal_mean[3])
    TSm = np.array([TSmm[0],TSmm[5],TSmm[1]+TSmm[2],TSmm[7],TSmm[9],TSmm[10],TSmm[11]])
    TScvt = np.array([TSmstd[0],TSmstd[5],TSmstd[1]+TSmstd[2],TSmstd[7],TSmstd[9],TSmstd[10],TSmstd[11]])
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
    ACC[:,fid] = np.array([max(r2_vod),max(r2_et),max(r2_sm),(p50_pct[2]-p50_pct[0])/7.5,np.nanpercentile(Geweke,25)])
    
    # summary_name = './Summary_'+TAG+'.pkl'
    # with open(summary_name,'wb') as f: pickle.dump((BIAS,CVE,ACC,mLAI),f)
       


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
plt.yticks(np.arange(len(acclist))+0.5,acclist,rotation=0)
plt.title('Accuracy and flatness')  
plt.xlabel('Pixels')


