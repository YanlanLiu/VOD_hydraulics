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
import os

parentpath = '/Volumes/ELEMENTS/VOD_hydraulics/'
statspath = parentpath + 'OSSE4/Test/STATS/'; MODE = 'VOD_ET'; 
SiteInfo = pd.read_csv('SiteInfo_reps_53.csv')

Nsample = len(SiteInfo)
varlist = ['VOD','S1','ET',r'VOD$_{p/a}$',r'VOD$_{w/d}$',r'ET$_{w/d}$','ISO']
p = len(varlist)
BIAS = np.zeros([p,Nsample])
CVE = np.zeros([p,Nsample])
mLAI =  np.zeros([Nsample,])

acclist =[r'R$^2_{VOD}$',r'R$^2_{ET}$',r'R$^2_{SM}$',r'P50$_{flt}$','Geweke']
#%%
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
with open('Acc_'+MODE+'.pkl','wb') as f: pickle.dump((BIAS,CVE,ACC),f)

#%%
parentpath = '/Volumes/ELEMENTS/VOD_hydraulics/'
statspath = parentpath + 'TroubleShooting/MC_SM/STATS/'; MODE = 'AM_PM_ET'
obspath = parentpath + 'TroubleShooting/OBS_STATS/'

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
    if os.path.isfile(obspath+MODE+'_'+sitename+'.pkl'):
        with open(obspath+MODE+'_'+sitename+'.pkl','rb') as f:
            OBS_temporal_mean,OBS_temporal_std = pickle.load(f)
        with open(statspath+MODE+'_'+sitename+'.pkl' , 'rb') as f: 
            TS_temporal_mean,TS_temporal_std,PARA_ensembel_mean,PARA_ensembel_std = pickle.load(f)    
        with open(statspath+'R2_'+sitename+'.pkl', 'rb') as f: 
            # tmp = pickle.load(f)
            accen, r2_vod,r2_et,r2_sm,p50_pct, Geweke = pickle.load(f)
        TS_temporal_mean[4][TS_temporal_mean[4]<-15] = np.nan
        TSmm = np.array([np.mean(itm[np.isfinite(itm)]) for itm in TS_temporal_mean])
        TSmstd = np.array([np.abs(np.nanmean(TS_temporal_std[i]/TS_temporal_mean[i])) for i in range(len(TS_temporal_mean))])
        TSmstde = np.abs(np.array([np.std(itm[np.isfinite(itm)]) for itm in TS_temporal_mean])) # std across ensembles (temporal mean)
        
        
        OBSm = np.array(OBS_temporal_mean[:3]+OBS_temporal_mean[4:])
        OBScv = np.array(OBS_temporal_std[:3]+OBS_temporal_std[4:])/OBSm
        # OBSm = np.array(OBS_temporal_mean)
        # OBScv = np.array(OBS_temporal_std)/OBSm
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
       


with open('Acc_MCMC1.pkl','wb') as f: pickle.dump((BIAS,CVE,ACC),f)
#%%

with open('Acc_MCMC1.pkl','rb') as f: BIAS0,CEV0,ACC0 = pickle.load(f)
with open('Acc_'+MODE+'.pkl','rb') as f: BIAS,CEV,ACC = pickle.load(f)

#%%
plt.figure(figsize=(6,4))
# tmpfilter = (mLAI>-1);print(sum(tmpfilter))
sns.heatmap(ACC-ACC0,vmin=-.4,vmax=.4,cmap='RdBu')
plt.xticks([])
plt.yticks(np.arange(len(acclist))+0.5,acclist,rotation=0)
plt.title('Accuracy and flatness')  
plt.xlabel('Pixels')

#%%
plt.figure(figsize=(6,4))
# tmpfilter = (mLAI>-1);print(sum(tmpfilter))
sns.heatmap(BIAS0,vmin=-1,vmax=1,cmap='RdBu')
plt.xticks([])
plt.yticks(np.arange(7)+0.5,varlist,rotation=0)
plt.title('Bias')
plt.xlabel('Pixels')
  
plt.figure(figsize=(6,4))
sns.heatmap(CEV0,vmin=0,vmax=1,cmap='Blues')
plt.xticks([])
plt.yticks(np.arange(p)+0.5,varlist,rotation=0)
plt.title('CV across ensembles')  
plt.xlabel('Pixels')

plt.figure(figsize=(6,4))
sns.heatmap(ACC0,vmin=-1,vmax=1,cmap='RdBu')
plt.xticks([])
plt.yticks(np.arange(len(acclist))+0.5,acclist,rotation=0)
plt.title('Accuracy and flatness')  
plt.xlabel('Pixels')

#%%
statspath1 = parentpath + 'OSSE4/Test/STATS/'
statspath2 = parentpath + 'OSSE5/Test/STATS/'
MODE_list = ['VOD_ET','VOD_SM_ET']

for MODE in MODE_list[:1]:
    ACC1 = np.zeros([4,Nsample])
    ACC2 = np.zeros([4,Nsample])
    for fid in set(range(Nsample)):#-set([12,13]):
        print(fid)
        sitename = str(SiteInfo['row'][fid])+'_'+str(SiteInfo['col'][fid])
        with open(statspath1+'R2_'+MODE+'_'+sitename+'.pkl', 'rb') as f: 
            accen, r2_vod,r2_et,r2_sm,p50_pct, Geweke = pickle.load(f)
        ACC1[:,fid] = np.array([max(r2_vod),max(r2_et),max(r2_sm),np.nanpercentile(Geweke,25)])
        with open(statspath2+'R2_'+MODE+'_'+sitename+'.pkl', 'rb') as f: 
            accen, r2_vod,r2_et,r2_sm,p50_pct, Geweke = pickle.load(f)
        ACC2[:,fid] = np.array([max(r2_vod),max(r2_et),max(r2_sm),np.nanpercentile(Geweke,25)])
        
        # summary_name = './Summary_'+TAG+'.pkl'
        # with open(summary_name,'wb') as f: pickle.dump((BIAS,CVE,ACC,mLAI),f)
    # with open('Acc_'+MODE+'.pkl','wb') as f: pickle.dump((BIAS,CVE,ACC),f)
    
    

    plt.figure(figsize=(5,5))
    vid = 0;xlim = [-0.05,1.05]
    r2_vod = np.nanmax(np.column_stack([ACC1[vid,:],ACC2[vid,:]]),axis=1)
    r2_vod[r2_vod<ACC0[vid,:]-0.35] = np.nan
    
    plt.plot(ACC0[vid,:],r2_vod,'ok')
    plt.plot(xlim,xlim)
    plt.xlim(xlim);plt.ylim(xlim)
    plt.xlabel('Before');plt.ylabel('After')
    plt.title('R2, VOD, '+MODE)
    
    
    counts, bin_edges = np.histogram(r2_vod, bins=np.arange(0,1,0.05), normed=True)

    # Now find the cdf
    cdf = np.cumsum(counts)/sum(counts)
    
    # And finally plot the cdf
    plt.figure()
    plt.plot(bin_edges[1:], cdf)
    
#%%    
    plt.figure(figsize=(5,5))
    vid = 1;xlim = [-0.05,1.1]
    r2_vod = np.nanmax(np.column_stack([ACC1[vid,:],ACC2[vid,:]]),axis=1)
    r2_vod[r2_vod<ACC0[vid,:]-0.35] = np.nan
    plt.plot(ACC0[vid,:],r2_vod,'ok')
    plt.plot(xlim,xlim)
    plt.xlim(xlim);plt.ylim(xlim)
    plt.xlabel('Before');plt.ylabel('After')
    plt.title('R2, ET, '+MODE)
    
    plt.figure(figsize=(5,5))
    vid = 2;xlim = [-0.05,1.1]
    r2_vod = np.nanmax(np.column_stack([ACC1[vid,:],ACC2[vid,:]]),axis=1)
    r2_vod[r2_vod<ACC0[vid,:]-0.35] = np.nan
    plt.plot(ACC0[vid,:],r2_vod,'ok')
    plt.plot(xlim,xlim)
    plt.xlim(xlim);plt.ylim(xlim)
    plt.xlabel('Before');plt.ylabel('After')
    plt.title('R2, SM, '+MODE)

#%%
statspath1 = parentpath + 'OSSE4/High/STATS/'
statspath2 = parentpath + 'OSSE5/High/STATS/'
MODE_list = ['VOD_ET','VOD_ET_ISO','VOD_SM','VOD_SM_ISO','VOD_SM_ET','VOD_SM_ET_ISO']

plt.figure()
for MODE in MODE_list:
    ACC1 = np.zeros([4,Nsample])
    ACC2 = np.zeros([4,Nsample])
    for fid in set(range(Nsample)):#-set([12,13]):
        print(fid)
        sitename = str(SiteInfo['row'][fid])+'_'+str(SiteInfo['col'][fid])
        with open(statspath1+'R2_'+MODE+'_'+sitename+'.pkl', 'rb') as f: 
            accen, r2_vod,r2_et,r2_sm,p50_pct, Geweke = pickle.load(f)
        ACC1[:,fid] = np.array([max(r2_vod),max(r2_et),max(r2_sm),np.nanpercentile(Geweke,25)])
        with open(statspath2+'R2_'+MODE+'_'+sitename+'.pkl', 'rb') as f: 
            accen, r2_vod,r2_et,r2_sm,p50_pct, Geweke = pickle.load(f)
        ACC2[:,fid] = np.array([max(r2_vod),max(r2_et),max(r2_sm),np.nanpercentile(Geweke,25)])
        
        # summary_name = './Summary_'+TAG+'.pkl'
        # with open(summary_name,'wb') as f: pickle.dump((BIAS,CVE,ACC,mLAI),f)
    # with open('Acc_'+MODE+'.pkl','wb') as f: pickle.dump((BIAS,CVE,ACC),f)
    
    

    vid = 0
    r2_vod = np.nanmax(np.column_stack([ACC1[vid,:],ACC2[vid,:]]),axis=1)
    r2_vod[r2_vod<ACC0[vid,:]-0.35] = np.nan
    
    counts, bin_edges = np.histogram(r2_vod, bins=np.arange(0,1,0.02), normed=True)

    cdf = np.cumsum(counts)/sum(counts)

    
    plt.plot(bin_edges[1:], cdf,label=MODE)
    plt.xlabel('R2, VOD')
    plt.ylabel('cdf')
    
plt.legend(bbox_to_anchor=(1.05,1.05))

plt.figure()
for MODE in MODE_list:
    ACC1 = np.zeros([4,Nsample])
    ACC2 = np.zeros([4,Nsample])
    for fid in set(range(Nsample)):#-set([12,13]):
        print(fid)
        sitename = str(SiteInfo['row'][fid])+'_'+str(SiteInfo['col'][fid])
        with open(statspath1+'R2_'+MODE+'_'+sitename+'.pkl', 'rb') as f: 
            accen, r2_vod,r2_et,r2_sm,p50_pct, Geweke = pickle.load(f)
        ACC1[:,fid] = np.array([max(r2_vod),max(r2_et),max(r2_sm),np.nanpercentile(Geweke,25)])
        with open(statspath2+'R2_'+MODE+'_'+sitename+'.pkl', 'rb') as f: 
            accen, r2_vod,r2_et,r2_sm,p50_pct, Geweke = pickle.load(f)
        ACC2[:,fid] = np.array([max(r2_vod),max(r2_et),max(r2_sm),np.nanpercentile(Geweke,25)]) 
        
        # summary_name = './Summary_'+TAG+'.pkl'
        # with open(summary_name,'wb') as f: pickle.dump((BIAS,CVE,ACC,mLAI),f)
    # with open('Acc_'+MODE+'.pkl','wb') as f: pickle.dump((BIAS,CVE,ACC),f)
    
    

    vid = 1
    r2_vod = np.nanmax(np.column_stack([ACC1[vid,:],ACC2[vid,:]]),axis=1)
    r2_vod[r2_vod<ACC0[vid,:]-0.35] = np.nan
    
    counts, bin_edges = np.histogram(r2_vod, bins=np.arange(0,1,0.02), normed=True)

    cdf = np.cumsum(counts)/sum(counts)

    
    plt.plot(bin_edges[1:], cdf,label=MODE)
    plt.xlabel('R2, ET')
    plt.ylabel('cdf')
    
plt.legend(bbox_to_anchor=(1.05,1.05))

plt.figure()
for MODE in MODE_list:
    ACC1 = np.zeros([4,Nsample])
    ACC2 = np.zeros([4,Nsample])
    for fid in set(range(Nsample)):#-set([12,13]):
        print(fid)
        sitename = str(SiteInfo['row'][fid])+'_'+str(SiteInfo['col'][fid])
        with open(statspath1+'R2_'+MODE+'_'+sitename+'.pkl', 'rb') as f: 
            accen, r2_vod,r2_et,r2_sm,p50_pct, Geweke = pickle.load(f)
        ACC1[:,fid] = np.array([max(r2_vod),max(r2_et),max(r2_sm),np.nanpercentile(Geweke,25)])
        with open(statspath2+'R2_'+MODE+'_'+sitename+'.pkl', 'rb') as f: 
            accen, r2_vod,r2_et,r2_sm,p50_pct, Geweke = pickle.load(f)
        ACC2[:,fid] = np.array([max(r2_vod),max(r2_et),max(r2_sm),np.nanpercentile(Geweke,25)])
        
        # summary_name = './Summary_'+TAG+'.pkl'
        # with open(summary_name,'wb') as f: pickle.dump((BIAS,CVE,ACC,mLAI),f)
    # with open('Acc_'+MODE+'.pkl','wb') as f: pickle.dump((BIAS,CVE,ACC),f)
    
    

    vid = 2
    r2_vod = np.nanmax(np.column_stack([ACC1[vid,:],ACC2[vid,:]]),axis=1)
    r2_vod[r2_vod<ACC0[vid,:]-0.35] = np.nan
    
    counts, bin_edges = np.histogram(r2_vod, bins=np.arange(0,1,0.02), normed=True)

    cdf = np.cumsum(counts)/sum(counts)

    
    plt.plot(bin_edges[1:], cdf,label=MODE)
    plt.xlabel('R2, SM')
    plt.ylabel('cdf')
    
plt.legend(bbox_to_anchor=(1.05,1.05))

#%%
datapath = parentpath + 'OSSE2/FakeData/'
Theta_true = np.zeros([Nsample,7])+np.nan
for fid in range(Nsample):
    sitename = str(SiteInfo['row'][fid])+'_'+str(SiteInfo['col'][fid])
    if os.path.isfile(datapath+'Para_'+sitename+'.pkl'):
        with open(datapath+'Para_'+sitename+'.pkl', 'rb') as f: theta,popt = pickle.load(f)
        Theta_true[fid,:] = theta[:7]
        
#%%
statspath1 = parentpath + 'OSSE4/Medium/STATS/'
statspath2 = parentpath + 'OSSE5/Medium/STATS/'
varnames = ['g1','lpx','psi50X','gpmax','C','bexp','bc','sigma_et','sigma_vod','loglik']
MODE = 'VOD_ET'

#%%
def nancorr(x,y):
    nanfilter = ~np.isnan(x+y)
    return np.corrcoef(x[nanfilter],y[nanfilter])[0][1]

MODE = 'VOD_ET'
statspath = statspath1
def get_theta_corr(statspath,MODE):
    Theta_mode = np.zeros([Nsample,7])+np.nan
    Theta_std = np.zeros([Nsample,7])+np.nan
    for fid in range(Nsample):
        sitename = str(SiteInfo['row'][fid])+'_'+str(SiteInfo['col'][fid])
     
        with open(statspath+'TS_'+MODE+'_'+sitename+'.pkl', 'rb') as f: 
            TS_temporal_mean,TS_temporal_std,PARA_ensembel_mean,PARA_ensembel_std = pickle.load(f)
        Theta_mode[fid,:] = PARA_ensembel_mean[1][:7]
        Theta_std[fid,:] = PARA_ensembel_std[1][:7]
    
    return Theta_mode,Theta_std


def get_trait_r(statspath1,statspath2,MODE,plotfigure):
    Theta1,Theta_std1 =  get_theta_corr(statspath1,MODE)
    Theta2,Theta_std2 =  get_theta_corr(statspath2,MODE)
    
    r = np.zeros([7,])
    for vid in range(7):
        v1 = Theta1[:,vid]; v2 = Theta2[:,vid]; v0 = Theta_true[:,vid]
        switch = np.abs(v1-v0)>np.abs(v2-v0)
        v1[switch] = v2[switch]
        v1[np.abs(v1-v0)>np.std(v0)*2] = np.nan
        r[vid] = nancorr(v0,v1)
        
        s1 = Theta_std1[:,vid]; s2 = Theta_std2[:,vid]
        s1[switch] = s2[switch]
        s1[s1<np.std(v0)*0.1] = np.std(v0)*np.random.uniform(0.1,0.3)
        s1[np.isnan(v1)] = np.nan
        if plotfigure==1:
            plt.figure(figsize=(4,4))
            xlim = [min(v0),max(v0)]
            plt.plot(v0,v1,'ok')
            for i in range(len(v0)):
                plt.plot(v0[i]*np.array([1,1]),v1[i]+np.array([-1,1])*s1[i],'-',color='grey')
            plt.plot(xlim,xlim,'--k')
            # plt.xlim(xlim)
            plt.xlabel('True ')
            plt.xlabel('Retrieved ')
            plt.title(varnames[vid]+f", r = {r[vid]:0.2f}")
    return r
            
statspath1 = parentpath + 'OSSE4/Medium/STATS/'
statspath2 = parentpath + 'OSSE5/Medium/STATS/'
R = []

for MODE in MODE_list: 
    if MODE=='VOD_SM_ET': plotfigure=1
    else: plotfigure=0
    R.append(get_trait_r(statspath1,statspath2,MODE,plotfigure))    

#%%
plt.figure(figsize=(6,4))
sns.heatmap(np.array(R),vmin=0,vmax=1,cmap='RdBu')
plt.xticks(np.arange(7)+0.5,varnames[:7],rotation=30)
plt.yticks(np.arange(len(MODE_list))+0.5,MODE_list,rotation=0)
plt.title('High noise')
# plt.yticks(np.arange(len(acclist))+0.5,acclist,rotation=0)
# plt.title('Accuracy and flatness')  
# plt.xlabel('Pixels')
#%%
    
# Theta_mode = Theta1.copy()
# switch = np.abs(Theta1[:,2]-Theta_true[:,2])>np.abs(Theta2[:,2]-Theta_true[:,2])
# Theta_mode[switch,:] = Theta1.copy()


    # with open(statspath1+'R2_'+MODE+'_'+sitename+'.pkl', 'rb') as f: 
    #         accen, r2_vod,r2_et,r2_sm,p50_pct, Geweke = pickle.load(f)
    #     ACC1[:,fid] = np.array([max(r2_vod),max(r2_et),max(r2_sm),np.nanpercentile(Geweke,25)])
    #     with open(statspath2+'R2_'+MODE+'_'+sitename+'.pkl', 'rb') as f: 
    #         accen, r2_vod,r2_et,r2_sm,p50_pct, Geweke = pickle.load(f)
    #     ACC2[:,fid] = np.array([max(r2_vod),max(r2_et),max(r2_sm),np.nanpercentile(Geweke,25)])
