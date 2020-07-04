#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 15:50:49 2020

@author: yanlan

Trade offs between P50 and model states
"""


import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="ticks", color_codes=True,font_scale=1.5)
import warnings; warnings.simplefilter("ignore")
import sys; sys.path.append("../Utilities/")
from newfun import readCLM, varnames, GetTrace
from Utilities import MovAvg, nancoef
from scipy.stats import norm


def CenterScale(x):
    # std = np.std(x,axis=0); 
    # mm = np.mean(x,axis=0)
    # std[std==0] = mm[std==0]
    x = np.apply_along_axis(lambda a: a/np.std(a),0,x)
    x[~np.isfinite(x)] = 1
    n,p = x.shape
    C = np.eye(n)-np.ones([n,n])/n
    x = np.dot(C,x)
    return x

def PCA(df):
    x = df.replace([np.inf, -np.inf], np.nan).dropna().values
    x = CenterScale(x)
    S = np.dot(np.transpose(x),x)
    w,v = np.linalg.eig(S)
    return w,v



parentpath = '/Volumes/ELEMENTS/VOD_hydraulics/'
TAG = 'WT_ISO'
versionpath = parentpath + 'TroubleShooting/'+TAG+'/'
inpath = parentpath+ 'Input/'
outpath = versionpath+'Output/'
forwardpath = versionpath+'Forward/'
statspath = versionpath+'STATS/'
obspath = versionpath+'../OBS_STATS/'
MODE = 'AM_PM_ET'
SiteInfo = pd.read_csv('SiteInfo_reps.csv')

R2 = np.zeros([100,2])
idx_sigma_vod = varnames.index('sigma_vod'); idx_sigma_et = varnames.index('sigma_et')

tslist = ['VOD','E','T','ET_AP','PSIL','S1','S2','VODr_ampm','ETr_ampm','VODr_wd','ETr_wd','ISO']
tslist_short = ['VOD','T','PSIL','S1','VODr_ampm','VODr_wd','ETr_wd','ISO']
accnames = ['r2_vod','r2_et','r2_sm','loglik_vod','loglik_et']

V1_state = np.zeros([len(tslist_short)+1,100])
V2_state = np.zeros([len(tslist_short)+1,100])
corr_state = np.zeros([len(tslist_short),100])

V1_acc = np.zeros([len(accnames)+1,100])
V2_acc = np.zeros([len(accnames)+1,100])
corr_acc = np.zeros([len(accnames),100])

V1_para = np.zeros([len(varnames)+3,100])
V2_para = np.zeros([len(varnames)+3,100])
corr_para = np.zeros([len(varnames)+2,100])

for fid in range(0,100):
    print(fid)
    sitename = str(SiteInfo['row'][fid])+'_'+str(SiteInfo['col'][fid])  
    
    # PREFIX = outpath+MODE+'_'+sitename+'_'
    # trace = GetTrace(PREFIX,0,optimal=False)
    # for v in varnames:
    #     plt.figure();plt.plot(trace[v][:]);plt.ylabel(v)
        
    with open(statspath+MODE+'_'+sitename+'.pkl' , 'rb') as f: 
        TS_temporal_mean,TS_temporal_std,PARA_ensembel_mean,PARA_ensembel_std = pickle.load(f)  

       
    with open(statspath+'R2_'+sitename+'.pkl', 'rb') as f: 
        enacc,r2_vod,r2_et,r2_sm,p50_pct,Geweke = pickle.load(f)
        
    with open(forwardpath+MODE+'_'+sitename+'.pkl', 'rb') as f: 
        TS,PARA = pickle.load(f)
        
    with open(obspath+MODE+'_'+sitename+'.pkl', 'rb') as f: 
        OBS_temporal_mean,OBS_temporal_std = pickle.load(f)
    
    # VOD,SOILM,ET,VODr_ampm,VODr_wd,ETr_wd,ISO = OBS_temporal_mean or OBS_temporal_std
    # OBS_temporal_mean = OBS_temporal_mean[0,2,1,]

        
    Forcings,VOD,SOILM,ET,dLAI,discard_vod,discard_et,idx = readCLM(inpath,sitename)
    VOD_ma = np.reshape(VOD,[-1,2])
    VOD_ma = np.reshape(np.column_stack([MovAvg(VOD_ma[:,0],4),MovAvg(VOD_ma[:,1],4)]),[-1,])
    
    valid_vod = ~np.isnan(VOD_ma)
    valid_et = ~np.isnan(ET)
    
    loglik_et = np.zeros([TS[0].shape[0],])
    loglik_vod = np.zeros([TS[0].shape[0],])

    for i in range(TS[0].shape[0]):
        loglik_vod[i] = np.nanmean(norm.logpdf(VOD_ma[valid_vod],TS[0][i,valid_vod],PARA[1][i,idx_sigma_vod]))
        loglik_et[i] = np.nanmean(norm.logpdf(ET[valid_et],(TS[1]+TS[2])[i,valid_et],PARA[1][i,idx_sigma_et]))
    
    P50 = PARA[1][:,2]
    tmpfilter = (PARA[1][:,-1]>np.percentile(PARA[1][:,-1],50))*(~np.isnan(P50))
    
    P50 = PARA[1][:,2]
    df = pd.DataFrame(np.column_stack([P50,np.transpose(np.array(TS_temporal_mean))]),columns=['P50']+tslist)[['P50']+tslist_short].loc[tmpfilter,:]
    w,v = PCA(df)
    V1_state[:,fid],V2_state[:,fid] = (v[:,0],v[:,1])
    # corr_state[:,fid] = np.corrcoef(np.transpose(df))[0,1:]
    corr_state[:,fid] = np.array([nancoef(df.values[:,0],df.values[:,i]) for i in range(1,len(list(df)))])
    
    
    df = pd.DataFrame(np.column_stack([P50,r2_vod,r2_et,r2_sm,loglik_vod,loglik_et]),columns=['P50']+accnames).loc[tmpfilter,:]
    w,v = PCA(df)
    V1_acc[:,fid],V2_acc[:,fid] = (v[:,0],v[:,1])
    corr_acc[:,fid] = np.array([nancoef(df.values[:,0],df.values[:,i]) for i in range(1,len(list(df)))])
    
    tmpidx = [itm!='psi50X' for itm in varnames]
    df = pd.DataFrame(np.column_stack([P50,PARA[0],PARA[1][:,tmpidx]]),columns=['P50','a','b','c']+[itm for itm in varnames if itm!='psi50X' ]).loc[tmpfilter,:]
    w,v = PCA(df)
    V1_para[:,fid],V2_para[:,fid] = (v[:,0],v[:,1])
    corr_para[:,fid] = np.array([nancoef(df.values[:,0],df.values[:,i]) for i in range(1,len(list(df)))])
    
# plt.plot(loglik_vod+loglik_et,PARA[1][:,-1],'o')

#%%
plt.figure(figsize=(6,4))
tmpfilter = np.load('tmpfilter.npy'); #tmpfilter[tmpfilter==False] = True
# print(sum(tmpfilter))
sns.heatmap(corr_state[:,tmpfilter],vmin=-1,vmax=1,cmap='RdBu')
plt.xticks([])
plt.yticks(np.arange(len(tslist_short))+0.5,tslist_short,rotation=0)
plt.title('Corr. with P50')
plt.xlabel('Pixels')
    

plt.figure(figsize=(6,4))
sns.heatmap(corr_acc[:,tmpfilter],vmin=-1,vmax=1,cmap='RdBu')
plt.xticks([])
plt.yticks(np.arange(len(accnames))+0.5,accnames,rotation=0)
plt.title('Corr. with P50')
plt.xlabel('Pixels')

plt.figure(figsize=(6,4))
sns.heatmap(corr_para[:,tmpfilter],vmin=-1,vmax=1,cmap='RdBu')
plt.xticks([])
plt.yticks(np.arange(len(varnames)+2)+0.5,['a','b','c']+[itm for itm in varnames if itm!='psi50X' ],rotation=0)
plt.title('Corr. with P50')
plt.xlabel('Pixels')
    
 
#%%
tmp = (PARA[0][:,0]>0)
plt.plot(P50[tmp],PARA[0][tmp,0],'ok')


#%%
# PARA.shape
# plt.plot(PARA[1][:,2],PARA[1][:,0],'o')#;plt.xscale('log');
# VOD,E,T,ET_AP,PSIL,S1,S2,VODr_ampm, ETr_ampm, VODr_wd, ETr_wd = TS_temporal_mean

for i in range(2,3):#range(len(TS_temporal_mean)):
    plt.figure(figsize=(4,4))
    loglik = PARA[1][:,-1]
    plt.plot(PARA[1][:,2],TS_temporal_mean[i],'o')#;plt.xscale('log');
    plt.xlabel('P50')
    plt.ylabel(tslist[i])
    tmp = TS_temporal_mean[i]
    # plt.plot([min(PARA[1][:,2]),max(PARA[1][:,2])],OBS_temporal_mean[-1]*np.array([1,1]))

    
#%% PCA


df = pd.DataFrame(np.column_stack([PARA[1][:,2],np.transpose(np.array(TS_temporal_mean))]),columns=['P50']+tslist)
df = df.replace([np.inf, -np.inf], np.nan).dropna()
x = df.loc[:, ['P50','VOD','T','PSIL','S1','VODr_ampm','VODr_wd','ETr_wd','ISO']].values
x = CenterScale(x)

S = np.dot(np.transpose(x),x)
w,v = np.linalg.eig(S)
F = np.dot(x,v)
for i,vname in enumerate(['P50','VOD','T','PSIL','S1','VODr_ampm','VODr_wd','ETr_wd','ISO']):
    plt.plot(v[i,0],v[i,1],'ok')
    plt.text(v[i,0],v[i,1],vname)
plt.xlabel('PC1')
plt.ylabel('PC2')


#%%
accnames = ['r2_vod','r2_et','r2_sm','ll_vod','ll_et']
df = pd.DataFrame(np.column_stack([PARA[1][:,2],r2_vod,r2_et,r2_sm,loglik_vod,loglik_et]),columns=['P50']+accnames)
df = df.replace([np.inf, -np.inf], np.nan).dropna()
x = df.loc[:, :].values
x = CenterScale(x)

S = np.dot(np.transpose(x),x)
w,v = np.linalg.eig(S)
F = np.dot(x,v)
for i,vname in enumerate(['P50']+accnames):
    plt.plot(v[i,0],v[i,1],'ok')
    plt.text(v[i,0],v[i,1],vname)
plt.xlabel('PC1')
plt.ylabel('PC2')


#%%
df = pd.DataFrame(np.column_stack([PARA[0],PARA[1]]),columns=['a','b','c']+varnames)
df = df.replace([np.inf, -np.inf], np.nan).dropna()
x = df.loc[:, :].values
x = CenterScale(x)

S = np.dot(np.transpose(x),x)
w,v = np.linalg.eig(S)
F = np.dot(x,v)
for i,vname in enumerate(['a','b','c']+varnames):
    plt.plot(v[i,0],v[i,1],'ok')
    plt.text(v[i,0],v[i,1],vname)
plt.xlabel('PC1')
plt.ylabel('PC2')

#%%
plt.plot(PARA[1][:,2],PARA[0][:,0],'ok')

#%%
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# x0 = StandardScaler().fit_transform(x)
# pca = PCA(n_components=3)
# principalComponents = pca.fit_transform(x)
# principalDf = pd.DataFrame(data = principalComponents
#              , columns = ['PC1', 'PC2','PC3'])
# finalDf = pd.concat([principalDf, df], axis = 1)
# print(pca.explained_variance_ratio_)

# fig = plt.figure(figsize = (8,8))
# ax = fig.add_subplot(1,1,1) 
# ax.set_xlabel('Principal Component 1', fontsize = 15)
# ax.set_ylabel('Principal Component 2', fontsize = 15)
# ax.set_title('2 component PCA', fontsize = 20)
# targets = ['P50', 'VOD', 'T']
# colors = ['r', 'g', 'b']
# for target, color in zip(targets,colors):
#     # indicesToKeep = finalDf['target'] == target
#     ax.scatter(finalDf['PC1'], finalDf['PC2'], s = 50)
# ax.legend(targets)
# ax.grid()

#%%
S = np.dot(np.transpose(x),x)
w,v = np.linalg.eig(S)
F = np.dot(x,v)
for i,vname in enumerate(['P50','VOD','T','PSIL','S1','VODr_ampm','VODr_wd','ETr_wd','ISO']):
    plt.plot(v[i,0],v[i,1],'ok')
    plt.text(v[i,0],v[i,1],vname)
plt.xlabel('PC1')
plt.ylabel('PC2')
# plt.plot(-F[:,0],principalComponents[:,0])
