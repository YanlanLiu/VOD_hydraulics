#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 23:03:35 2020

@author: yanlan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 20:08:17 2020

@author: yanlan
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns; sns.set(style="ticks", color_codes=True,font_scale=1.5)
import os


parentpath = '/Volumes/ELEMENTS/VOD_hydraulics/'
obspath = parentpath + 'TroubleShooting/OBS_STATS/'

SiteInfo = pd.read_csv('SiteInfo_reps_53.csv')
Nsample = len(SiteInfo)
#%% OSSE_dET, R2
statspath1 = parentpath + 'OSSE4/Medium/STATS/'
# statspath2 = parentpath + 'OSSE5/Medium/STATS/'
statspath3 = parentpath + 'OSSE_ND/STATS/'
MODE_list = ['VOD_ET','VOD_SM_ET']
linetype = ['-','--']

for mid,MODE in enumerate(MODE_list):
    plt.figure()
    ACC1 = np.zeros([4,Nsample])+np.nan
    ACC2 = np.zeros([4,Nsample])+np.nan
    ACC3 = np.zeros([4,Nsample])+np.nan

    for fid in set(range(Nsample)):#-set([12,13]):
        print(fid)
        sitename = str(SiteInfo['row'][fid])+'_'+str(SiteInfo['col'][fid])
        with open(statspath1+'R2_'+MODE+'_'+sitename+'.pkl', 'rb') as f: 
            accen, r2_vod,r2_et,r2_sm,p50_pct, Geweke = pickle.load(f)
        ACC1[:,fid] = np.array([np.nanmax(r2_vod),np.nanmax(r2_et),np.nanmax(r2_sm),np.nanpercentile(Geweke,25)])
        # with open(statspath2+'R2_'+MODE+'_'+sitename+'.pkl', 'rb') as f: 
        #     accen, r2_vod,r2_et,r2_sm,p50_pct, Geweke = pickle.load(f)
        # ACC2[:,fid] = np.array([max(r2_vod),max(r2_et),max(r2_sm),np.nanpercentile(Geweke,25)])
        
        fname = statspath3+'R2_'+MODE+'_'+sitename+'.pkl'
        if os.path.isfile(fname):
            with open(statspath3+'R2_'+MODE+'_'+sitename+'.pkl', 'rb') as f: 
                accen, r2_vod,r2_et,r2_sm,p50_pct, Geweke = pickle.load(f)
            ACC3[:,fid] = np.array([np.nanmax(r2_vod),np.nanmax(r2_et),np.nanmax(r2_sm),np.nanpercentile(Geweke,25)])
        
 
    vid = 1
    # r2_vod = np.nanmax(np.column_stack([ACC1[vid,:],ACC2[vid,:]]),axis=1)
    # # r2_vod[r2_vod<ACC0[vid,:]-0.35] = np.nan
    counts, bin_edges = np.histogram(ACC1[:,vid], bins=np.arange(0,1,0.02), normed=True)
    cdf = np.cumsum(counts)/sum(counts)
    plt.plot(bin_edges[1:], cdf,label='AMSRE_')
    counts, bin_edges = np.histogram(ACC3[:,vid], bins=np.arange(0,1,0.02), normed=True)
    cdf = np.cumsum(counts)/sum(counts)
    plt.plot(bin_edges[1:], cdf,label=MODE)
    # plt.plot(ACC1[vid,:],ACC3[vid,:],'o',label=MODE)
    
# plt.legend(bbox_to_anchor=(1.05,1.05))
# xlim = [0,1]
# plt.plot(xlim,xlim,'-k')
# plt.xlabel('Weekly ET');plt.ylabel('Daily ET')
# plt.title('R2, ET, medium noise')

#%% OSSE_dET, r

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
statspath0 = parentpath + 'OSSE4/Medium/STATS/'
varnames = ['g1','lpx','psi50X','gpmax','C','bexp','bc','sigma_et','sigma_vod','loglik']
MODE = 'VOD_ET'

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
        fname = statspath+'TS_'+MODE+'_'+sitename+'.pkl'
        if os.path.isfile(fname):
            with open(fname, 'rb') as f: 
                
                TS_temporal_mean,TS_temporal_std,PARA_ensembel_mean,PARA_ensembel_std = pickle.load(f)
        else:
            with open(statspath0+'TS_'+MODE+'_'+sitename+'.pkl', 'rb') as f: 
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
        v1[np.abs(v1-v0)>np.std(v0)*1.8] = np.nan
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

# statspath1 = parentpath + 'OSSE_dET/Medium/STATS/'
# statspath2 = parentpath + 'OSSE_dET/Medium/STATS/'

R = []

for MODE in MODE_list: 
    if MODE=='VOD_SM_ET': plotfigure=1
    else: plotfigure=0
    R.append(get_trait_r(statspath1,statspath2,MODE,plotfigure))    

plt.figure(figsize=(6,4))
sns.heatmap(np.array(R),vmin=0,vmax=1,cmap='RdBu')
plt.xticks(np.arange(7)+0.5,varnames[:7],rotation=30)
plt.yticks(np.arange(len(MODE_list))+0.5,MODE_list,rotation=0)
plt.title('Weekly ET, medium noise')
# plt.yticks(np.arange(len(acclist))+0.5,acclist,rotation=0)
# plt.title('Accuracy and flatness')  
# plt.xlabel('Pixels')