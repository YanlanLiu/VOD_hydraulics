#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 16:04:51 2020

@author: yanlan

"""


import numpy as np
import pandas as pd
import pickle
from datetime import timedelta,datetime
from newfun import GetTrace,nobsinaday, readCLM, hour2week, hour2day
# from myFun import varnames,lowbound,upbound,
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="ticks", color_codes=True,font_scale=1.75)
from Utilities import MovAvg, normalize,itp_rm_outlier,Fourier_transform
from newfun import varnames,lowbound,upbound
from newfun import start_date,end_date,fitVOD_RMSE

parentpath = '/Volumes/ELEMENTS/VOD_hydraulics/'
inpath = parentpath+'/Input/'
outpath = parentpath+'/Output/'
forwardpath = parentpath+'/Forward/'
SiteInfo = pd.read_csv('SiteInfo_US_full.csv')

fid = 1

sitename = str(SiteInfo['row'][fid])+'_'+str(SiteInfo['col'][fid])

Forcings,VOD,ET,dLAI,discard_vod,discard_et,idx = readCLM(inpath,sitename)

MODE = 'AM_PM_ET'
PREFIX = outpath+MODE+'_'+sitename+'_'
TITL =  MODE+'_'+sitename
print(TITL)
forwardname = forwardpath+MODE+'_'+sitename+'.pkl'
with open(forwardname,'rb') as f:  # Python 3: open(..., 'rb')
    SVOD, SET, SPSIL = pickle.load(f)



# trace_df = GetTrace(PREFIX,0,optimal=False)
# trace_df = trace_df[(trace_df['chain']==2)]
# plt.plot(trace_df['C'])

trace_df = GetTrace(PREFIX,0,optimal=False)
warmup = int(max(trace_df['step'])*0.8)
trace_df = trace_df[trace_df['step']>warmup].reset_index().drop(columns=['index'])    


ndays = (end_date-start_date).days
dPSIL = hour2day(np.array(SPSIL.mean(axis=0)),idx)[~discard_vod]
tt = np.array([start_date+timedelta(days = i*0.5) for i in range(ndays*2)])[~discard_vod]
DOY = np.array([itm.timetuple().tm_yday for itm in tt])
VOD_hat = np.mean(SVOD,axis=0)
tmp,popt = fitVOD_RMSE(dPSIL,dLAI,VOD,return_popt=True) 

a,b,c = popt
AGB = b+c*dLAI+c
RWC = VOD/AGB

plt.figure(figsize=(18,5))
plt.subplots_adjust(wspace=0.4)
plt.subplot(131)
plt.scatter(VOD,VOD_hat,c=DOY,cmap='RdBu')
plt.clim([0,365]); cbar = plt.colorbar()#cbar.set_label('DOY',rotation=270,labelpad=20)
xlim = [np.nanmin(VOD),np.nanmax(VOD)]
plt.plot(xlim,xlim,'--k')
plt.xlabel('Observed VOD');plt.ylabel('Modeled VOD')

plt.subplot(132)
dPSIL[dPSIL<-14] = np.nan
plt.plot(dPSIL,RWC,'o',alpha=0.3,label='VOD/AGB')
plt.plot(dPSIL,1+a*dPSIL,'o',alpha=0.3,label=r'$\overline{VOD}$/AGB')
plt.legend()
plt.xlabel(r'$\psi_l$ (MPa)'); plt.ylabel('RWC')



plt.figure(figsize=(12,12))
plt.subplot(311)
plt.plot(VOD,'o',color='grey',label='Observed')
plt.plot(VOD_hat,'or',label='Fitted')
plt.ylabel('VOD')
plt.legend(ncol=2)
plt.xticks([])
# plt.ylim(0.28,0.88)
tmpfilter = ~np.isnan(VOD+VOD_hat)
r2 = 1-np.sum((VOD_hat[tmpfilter]-VOD[tmpfilter])**2)/np.sum((VOD[tmpfilter]-np.mean(VOD[tmpfilter]))**2)
plt.title("R2 = %.2f" % r2)

ax21 = plt.subplot(312)
ax21.plot(RWC,'ob')
ax22 = ax21.twinx()
ax22.plot(AGB,'og')
ax21.set_xticks([])
ax21.set_ylabel('RWC',color='b')
ax22.set_ylabel('AGB',color='g')

plt.subplot(313)
plt.plot(ET,'o',color='grey',label='ALEXI')
plt.plot(np.min(SET,axis=0),'-',color='lightblue',label='Range')
plt.plot(np.max(SET,axis=0),'-',color='lightblue')
plt.plot(np.mean(SET,axis=0),'-',color='navy',label='Modeled')
plt.legend(ncol=3)
plt.ylabel("ET")
plt.xlabel('Time (week)')
tmpfilter = ~np.isnan(ET+np.mean(SET,axis=0))
r2 = 1-np.sum((np.mean(SET,axis=0)[tmpfilter]-ET[tmpfilter])**2)/np.sum((ET[tmpfilter]-np.mean(ET[tmpfilter]))**2)
plt.title("R2 = %.2f" % r2)


plt.figure(figsize=(15,15))
for i,var2plot in enumerate(varnames[:-1]):
    plt.subplot('33'+str(i+1))
    sns.kdeplot(trace_df[var2plot])
    # if i==4 or i==5: 
    #     plt.xscale('log')
    plt.xlim(lowbound[i],upbound[i])
