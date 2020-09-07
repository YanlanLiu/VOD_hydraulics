#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 13:26:15 2020

@author: yanlan
"""


import pickle
import os
import numpy as np
import pandas as pd
import warnings; warnings.simplefilter("ignore")
import sys; sys.path.append("../Utilities/")
from newfun import readCLM # newfun_full
from newfun import fitVOD_RMSE,dt, hour2day, hour2week
from newfun import get_var_bounds,OB,CONST,CLAPP,ca
from newfun import GetTrace
from Utilities import nanOLS,nancorr,MovAvg,IsOutlier
import time

tic = time.perf_counter()

# =========================== control pannel =============================

parentpath = '/scratch/users/yanlan/'
arrayid = int(os.environ['SLURM_ARRAY_TASK_ID']) # 0-935
nsites_per_id = 100
warmup, nsample,thinning = (0.8,10,100)

# parentpath = '/Volumes/ELEMENTS/VOD_hydraulics/'
# arrayid = 44#4672
# nsites_per_id = 100
# warmup, nsample,thinning = (0.8,2,40)

versionpath = parentpath + 'Global_0817/'
inpath = parentpath+ 'Input_Global/'
outpath = versionpath +'Output/'
forwardpath = versionpath+'Forward/'
statspath = versionpath+'STATS/'

MODE = 'VOD_SM_ET'
varnames, bounds = get_var_bounds(MODE)
SiteInfo = pd.read_csv('SiteInfo_globe_full.csv')

#%%

def calR2(yhat,y):
    return 1-np.nanmean((y-yhat)**2)/np.nanmean((y-np.nanmean(y))**2)

RSlist = [125,150,150,100,125,300,170,300,70,40,70,40,200,40,999,999,100,150,150,200]

OBS_mean = []; OBS_std = []; OBSnan = [np.nan for i in range(9)]
OBS_N = []; OBSNnan = [np.nan for i in range(3)]
TS_mean = []; TS_std = []; TSnan = [np.nan for i in range(4)]
PARA_mean = []; PARA_std = []; PARAnan = [np.nan for i in range(14)]
PARA2_mean = []; PARA2_std = []; PARA2nan = [np.nan for i in varnames]

ACC = []; ACCnan = [np.nan for i in range(4)]

for fid in range(arrayid*nsites_per_id,min((arrayid+1)*nsites_per_id,len(SiteInfo))):
    sitename = str(SiteInfo['row'].values[fid])+'_'+str(SiteInfo['col'].values[fid])
    print(sitename)
    try:
        Forcings,VOD,SOILM,ET,dLAI,discard_vod,discard_et,idx = readCLM(inpath,sitename)
        VOD[IsOutlier(VOD,multiplier=2)] = np.nan
        SOILM[IsOutlier(SOILM,multiplier=2)] = np.nan
    except FileNotFoundError as err:
        print(err)
        OBS_mean.append(OBSnan); OBS_std.append(OBSnan); OBS_N.append(OBSNnan)
        TS_mean.append(TSnan); TS_std.append(TSnan)
        PARA_mean.append(PARAnan); PARA_std.append(PARAnan)
        ACC.append(ACCnan)
        continue

    VOD_ma = np.reshape(VOD,[-1,2])
    VOD_ma = np.reshape(np.column_stack([MovAvg(VOD_ma[:,0],4),MovAvg(VOD_ma[:,1],4)]),[-1,])
    
    Z_r,tx = (SiteInfo['Root depth'].values[fid]*1000,int(SiteInfo['Soil texture'].values[fid]))
     
    psi0cm = CLAPP.psat[tx]
    phi0 = -psi0cm/100*9.8*1000/10**6 #MPa # *10**6/9.8 
    phi0_mm = -psi0cm*10 # mm
    n = CLAPP.thetas[tx]
    ksoil = CLAPP.ksat[tx]*60*10  #cm/s to mm/hr
    sinit = 0.28
    d1 = 50
    d2 = Z_r-d1
    m1 = -d1/2
    m2 = -(d1+d2/2)
    m3 = -(d1+d2+1000)
     
    # Calculations not affected by MCMC paramteres
    RNET,TEMP,P,VPD,Psurf,GA,LAI,VegK = Forcings
    N = len(RNET)
     
    # Terms in Farquhar's model of biochemical demand for CO2
    PAR = RNET/(CONST.Ephoton*CONST.NA)*1e6
    T_C = TEMP-CONST.U3 # degree C
    Kc = 300*np.exp(0.074*(T_C-25)) # umol/mol
    Ko = 300*np.exp(0.015*(T_C-25)) # mmol/mol
    cp = 36.9+1.18*(T_C-25)+0.036*(T_C-25)**2
    Vcmax25 = SiteInfo['Vcmax25'].values[fid]
    Vcmax0 = Vcmax25*np.exp(50*(TEMP-298)/(298*CONST.R*TEMP))
    Jmax = Vcmax0*np.exp(1)
    
    # Vcmax0 = OB.koptv*OB.Hdv*np.exp(OB.Hav*(TEMP-OB.Toptv)/TEMP/CONST.R/OB.Toptv)/(OB.Hdv-OB.Hav*(1-np.exp(OB.Hav*(TEMP-OB.Toptv)/TEMP/CONST.R/OB.Toptv)))
    # Jmax = OB.koptj*OB.Hdj*np.exp(OB.Haj*(TEMP-OB.Toptj)/TEMP/CONST.R/OB.Toptj)/(OB.Hdj-OB.Haj*(1-np.exp(OB.Haj*(TEMP-OB.Toptj)/TEMP/CONST.R/OB.Toptj))) 
    J = (OB.kai2*PAR+Jmax-np.sqrt((OB.kai2*PAR+Jmax)**2-4*OB.kai1*OB.kai2*PAR*Jmax))/2/OB.kai1
    
    # Terms in Penman-Monteith Equation
    VPD_kPa = VPD*Psurf
    sV = 0.04145*np.exp(0.06088*T_C) #in Kpa
    RNg = np.array(RNET*np.exp(-LAI*VegK))
    petVnum = (sV*(RNET-RNg)+1.225*1000*VPD_kPa*GA)*(RNET>0)/CONST.lambda0*60*60  #kg/s/m2/CONST.lambda0*60*60
    # (sV*(rnmg-1*RNgg) + 1.225*1000*myvpd*myga)*(myrn > 0)
    petVnumB = 1.26*(sV*RNg)/(sV+CONST.gammaV)/CONST.lambda0*60*60 
    
    PET = (sV*(RNET-RNg)+1.225*1000*VPD_kPa*GA)*(RNET>0)/CONST.lambda0*60*60/(sV+CONST.gammaV*(1+GA*RSlist[SiteInfo['IGBP'].iloc[fid]]))*24 # mm/day
    
    def advance_linearize(s2,phiL,ti,gpmax,C,psi50X,bexp,timestep):
        a = -1/(2*psi50X)
        # f_const = gpmax*(1+a*phiL)*(phi0*(s2/n)**(-bexp) - phiL)
        # f_x = gpmax*((a)*(phi0*(s2/n)**(-bexp) - phiL) + (1+a*phiL)*( - 1))
        # f_y = gpmax*(1+a*phiL)*(phi0*n**bexp*s2**(-bexp-1)*(-bexp))
        phiS2 = phi0*(s2/n)**(-bexp)
        delta_phi = phiS2 - phiL
     
        f_const = gpmax*(1+a*phiL)*delta_phi
        f_x = gpmax*(a*delta_phi + (1+a*phiL)*(-1))
        # f_y = gpmax*(1+a*phiL)*(phiS2*(-bexp)/s2-phiL) # need to double check
        f_y = gpmax*(1+a*phiL)*(phiS2*(-bexp)/s2) # need to double check
        # f_y = gpmax*(1+a*phiL)*(phi0*n**bexp*s2**(-bexp-1)*(-bexp))
        j0 = f_const - f_x*phiL - f_y*s2
        jp = f_x
        js = f_y
        k1 = jp/C - js/Z_r
        k0 = -jp/C*ti + k1*j0
        x0 = C*phiL + Z_r*s2
        xnew = -ti*timestep + x0
        y0 = jp*phiL + js*s2
        ynew = (y0 + k0/k1)*np.exp(k1*timestep) - k0/k1
        snew = (ynew - jp/C*xnew) / (-jp*Z_r/C + js)
        psiLnew = (xnew - Z_r*snew)/C
        return snew, psiLnew
     
    tdiv = 3
    def get_ti(clm,condS):
        RNET_i,a1_i,a2_i,Vcmax0_i,ci_i,LAI_i,petVnum_i,sV_i,GA_i = clm
        if condS>0 and RNET_i>0:
            An = max(0,min(a1_i*condS,a2_i)-0.015*Vcmax0_i*condS)
            gs = 1.6*An/(ca-ci_i)*LAI_i*0.02405
            ti = petVnum_i/(sV_i+CONST.gammaV*(1+GA_i*(1/GA_i+1/gs)))
        else: 
            ti = 0
        return ti
     
    def runhh_2soil_hydro(theta):    
        g1, lpx, psi50X, gpmax,C, bexp, sbot = theta[:7]
        
        medlyn_term = 1+g1/np.sqrt(VPD_kPa) # double check
        ci = ca*(1-1/medlyn_term)
        a1 = Vcmax0*(ci-cp)/(ci + Kc*(1+209/Ko))
        a2 = J*(ci-cp)/(4*(ci + 2*cp))
        
        psi50X = -1.*psi50X
        psi50L = lpx*psi50X
        # SP = SoilPara(SR[0],SR[1],bexp,sbot)
        
        p3 = phi0_mm*(sbot/n)**(-bexp)+m3 
        k3 = ksoil*(sbot/n)**(2*bexp) 
        
        phil_list = np.zeros([N,])
        et_list = np.zeros([N,])
        
        s1 = np.copy(sinit)
        s2 = np.copy(sinit) 
        phiL = phi0*(s2/n)**(-bexp) - 0.01
        
        s1_list = np.zeros([N,]); s2_list = np.zeros([N,])
        # e_list = np.zeros([N,]); t_list = np.zeros([N,])
        
        for i in np.arange(N):
    
            phil_list[i] = phiL*1.0
            clm = (RNET[i],a1[i],a2[i],Vcmax0[i],ci[i],LAI[i],petVnum[i],sV[i],GA[i])
            condS = max(min(1-phiL/(2*psi50L),1),0)
            ti = get_ti(clm,condS)
            s2_pred, phiL_pred = advance_linearize(s2,phiL,ti,gpmax,C,psi50X,bexp,dt)     
            if np.abs(phiL_pred-phiL) < np.abs(psi50L):
                s2 = np.copy(s2_pred)
                phiL = np.copy(phiL_pred)
            else:
                #print('tdiv...')
                tlist = np.zeros(tdiv)
                for subt in np.arange(tdiv):
                    condS = max(min(1-phiL/(2*psi50L),1),0)
                    tlist[subt] = get_ti(clm,condS)               
                    s2, phiL = advance_linearize(s2,phiL,tlist[subt],gpmax,C,psi50X,bexp,dt/tdiv)
                ti = np.mean(tlist)
            
            ei= petVnumB[i]*(s1/n) #**bexp#*s1/n#*soilfac#*((s1-smcwilt)/(n-smcwilt)**(1))
            s1 = min(s1+(P[i]-ei)*dt/d1,n)#p_e = P-E 
            
                  
            p1 = phi0_mm*(s1/n)**(-bexp) + m1
            p2 = phi0_mm*(s2/n)**(-bexp) + m2
            k1 = ksoil*(s1/n)**(2*bexp)
            k2 = ksoil*(s2/n)**(2*bexp)
            f12 = 2/(1/k1+1/k2) * (p1-p2) / (m1-m2)*dt
            f23 = 2/(1/k2+1/k3) * (p2-p3) / (m2-m3)*dt
            
            s1 = max(s1-f12/d1,0.05)
            s2 = min(max(s2+f12/d2 - f23/d2,0.05),n) 
            phiL = max(psi50X*2,phiL)
            
            s1_list[i] = np.copy(s1); s2_list[i] = np.copy(s2)
            et_list[i] = ei+ti
            
        s1_list[np.isnan(s1_list)] = np.nanmean(s1_list); s1_list[s1_list>1] = 1; s1_list[s1_list<0] = 0
        
        return phil_list,et_list,s1_list#,s2_list
        
    
    valid_sm = ~np.isnan(SOILM); SOILM_valid = SOILM[valid_sm]
    bins = np.arange(0,1.02,0.01)
    counts, bin_edges = np.histogram(SOILM_valid, bins=bins, normed=True)
    cdf1 = np.cumsum(counts)/sum(counts)
        
    PREFIX = outpath+MODE+'_'+sitename+'_'
    print(PREFIX)
    try:
        trace = GetTrace(PREFIX,0)
        trace = trace.sort_values(by=['loglik']).reset_index().drop(columns=['index']) 
        halftrace_mean = trace[int(len(trace)*0.5):].reset_index().drop(columns=['index'])[varnames].mean().values
        halftrace_std = trace[int(len(trace)*0.5):].reset_index().drop(columns=['index'])[varnames].std().values
        trace = trace[int(len(trace)*warmup):].reset_index().drop(columns=['index'])  
        
    except :
        OBS_mean.append(OBSnan); OBS_std.append(OBSnan); OBS_N.append(OBSNnan)
        TS_mean.append(TSnan); TS_std.append(TSnan)
        PARA_mean.append(PARAnan); PARA_std.append(PARAnan)
        PARA2_mean.append(PARA2nan); PARA2_std.append(PARA2nan); 
        ACC.append(ACCnan)
        continue
#    trace = trace[trace['step']>trace['step'].max()*warmup].reset_index().drop(columns=['index'])
          
    TS = [[] for i in range(4)]
    PARA = [[] for i in range(2)]
    
    for count in range(nsample):
        idx_s = max(len(trace)-1-count*thinning,0)#randint(0,len(trace))
        print(idx_s)
        try:
            tmp = trace['g1'].iloc[idx_s]
        except IndexError:
            print(trace)
            print(idx_s)
            print(sitename) 
        theta = trace.iloc[idx_s][varnames].values
        tic0 = time.perf_counter()
        PSIL_hat,ET_hat,S1_hat = runhh_2soil_hydro(theta)
        #toc0 = time.perf_counter()
        #print(f"net run time: {toc0-tic:0.4f} seconds")

        
        # ET_ampm = hour2day(E_hat+T_hat,[idx[1]-1,idx[1]])[~discard_vod]
        ET_hat = hour2week(ET_hat,UNIT=24)[~discard_et] # mm/hr -> mm/day
        dPSIL = hour2day(PSIL_hat,idx)[~discard_vod]
        VOD_hat,popt = fitVOD_RMSE(dPSIL,dLAI,VOD_ma,return_popt=True) 
        dS1 = hour2day(S1_hat,idx)[~discard_vod][::2]

        if np.isfinite(np.nansum(dS1)) and np.nansum(dS1)>0:
            counts, bin_edges = np.histogram(dS1, bins=bins, normed=True)
            cdf2 = np.cumsum(counts)/sum(counts)
            dS1_matched = np.array([bin_edges[np.abs(cdf1-cdf2[int(itm*100)]).argmin()] for itm in dS1])
        else:
            dS1_matched = np.zeros(dS1.shape)+np.nan
            

        TS = [np.concatenate([TS[ii],itm]) for ii,itm in enumerate((VOD_hat,ET_hat,PSIL_hat,dS1_matched))]
        PARA = [np.concatenate([PARA[ii],itm]) for ii,itm in enumerate((popt,theta))]
        
        #toc = time.perf_counter()
        #print(f"Sample time: {toc-tic:0.4f} seconds")

    TS = [np.reshape(itm,[nsample,-1]) for itm in TS] # VOD,ET,PSIL,S1
    PARA = [np.reshape(itm,[nsample,-1]) for itm in PARA]
        
       
    forwardname = forwardpath+'TS_'+MODE+'_'+sitename+'.pkl'
    with open(forwardname, 'wb') as f: pickle.dump(TS, f)
    forwardname = forwardpath+'PARA_'+MODE+'_'+sitename+'.pkl'
    with open(forwardname, 'wb') as f: pickle.dump(PARA, f)
    
    TS_temporal_mean = [np.nanmean(itm) for itm in TS] # temporal mean of ensemble mean
    TS_temporal_std = [np.nanstd(np.nanmean(itm,axis=0)) for itm in TS] # temporal std of ensemble mean
    PARA_ensembel_mean = np.concatenate([np.nanmean(itm,axis=0) for itm in PARA[::-1]])
    PARA_ensembel_std = np.concatenate([np.nanstd(itm,axis=0) for itm in PARA[::-1]])

        
    # ======== OBS stats ===========
    OBS = (VOD_ma,ET,SOILM,RNET,TEMP,P,VPD,PET,LAI)
    OBS_temporal_mean = [np.nanmean(itm) for itm in OBS]
    OBS_temporal_std = [np.nanstd(itm) for itm in OBS]
    obsn = [sum(~np.isnan(itm)) for itm in OBS[:3]]

    
    
    # ========= Convergence and perofrmance ==========
    sample_length = int(3e3); step = int(5e2)
    st_list = range(1000,int(len(trace)/sample_length)*sample_length-sample_length+1,step)
    Geweke = []
    varname= 'psi50X'
    chain = np.array(trace[varname])
    chain = chain[:int(len(chain)/sample_length)*sample_length] 
    for st in st_list:
        tmps = chain[st:(st+sample_length)]
        tmpe = chain[-sample_length:]
        Geweke.append((np.nanmean(tmps)-np.nanmean(tmpe))/np.sqrt(np.nanvar(tmps)+np.nanvar(tmpe)))
    Geweke = np.abs(np.array(Geweke))
    
    #r2_vod = np.apply_along_axis(nancorr,1,TS[0],VOD_ma)**2
    #r2_et = np.apply_along_axis(nancorr,1,TS[1],ET)**2
    #r2_sm = np.apply_along_axis(nancorr,1,TS[3],SOILM)**2
    er2_vod = nancorr(np.nanmean(TS[0],axis=0),VOD_ma)**2
    er2_et = nancorr(np.nanmean(TS[1],axis=0),ET)**2
    er2_sm = nancorr(np.nanmean(TS[3],axis=0),SOILM)**2
    acc_summary = [er2_vod,er2_et,er2_sm,np.nanpercentile(Geweke,25)]
    #print(acc_summary)
    #acc_summary = [np.nanmax(r2_vod),np.nanmax(r2_et),np.nanmax(r2_sm),np.nanpercentile(Geweke,25)]
    #print(acc_summary)
    
    OBS_mean.append(OBS_temporal_mean); OBS_std.append(OBS_temporal_std); OBS_N.append(obsn)
    TS_mean.append(TS_temporal_mean); TS_std.append(TS_temporal_std)
    PARA_mean.append(PARA_ensembel_mean); PARA_std.append(PARA_ensembel_std)
    PARA2_mean.append(halftrace_mean); PARA2_std.append(halftrace_std)
    ACC.append(acc_summary)
    toc = time.perf_counter()
    print(f"Site time: {toc-tic:0.4f} seconds")


#print(np.array(ACC))
#nn = int(len(ACC)/4);print(nn)
nn = np.array(ACC).shape[0];print(nn)
OBS_mean = np.reshape(np.array(OBS_mean),[nn,-1])
OBS_std = np.reshape(np.array(OBS_std),[nn,-1])
OBS_N = np.reshape(np.array(OBS_N),[nn,-1])

TS_mean = np.reshape(np.array(TS_mean),[nn,-1])
TS_std = np.reshape(np.array(TS_std),[nn,-1])
PARA_mean = np.reshape(np.array(PARA_mean),[nn,-1])
PARA_std = np.reshape(np.array(PARA_std),[nn,-1])
PARA2_mean = np.reshape(np.array(PARA2_mean),[nn,-1])
PARA2_std = np.reshape(np.array(PARA2_std),[nn,-1])
#ACC = np.reshape(np.array(ACC),[nn,-1])
ACC = np.array(ACC)
#print(PARA_mean)
#print(OBS_mean.shape,TS_mean.shape,PARA_mean.shape,ACC.shape)
#print(ACC)

obsname = statspath+'OBS_'+MODE+'_'+str(arrayid).zfill(3)+'.pkl'
with open(obsname, 'wb') as f: 
    pickle.dump((OBS_mean,OBS_std,OBS_N), f)
# OBS_mean, OBS_std = pickle.load(f)
# VOD_ma,ET,SOILM,RNET,TEMP,P,VPD,LAI,ISO = OBS_mean[sub_fid,:] # (OBS_std[sub_fid,:]) temporal mean (std) of observation
    
estname = statspath+'EST_'+MODE+'_'+str(arrayid).zfill(3)+'.pkl'
with open(estname, 'wb') as f: 
    pickle.dump((TS_mean,TS_std,PARA_mean,PARA_std,PARA2_mean,PARA2_std,ACC), f)
#TS_mean,TS_std,PARA_mean,PARA_std = pickle.load(f)
# VOD,ET,PSIL,S1 = TS_mean[sub_fid,:] # (TS_std[sub_fid,:]) temporal mean (std) of ensembel mean
# g1,lpx,psi50X,C,bexp,bc,sigma_et,sigma_vod,loglik,a,b,c = PARA_mean[sub_fid,:] # (PARA_std[sub_fid,:]) ensemble mean (std)
# r2_vod,r2_et,r2_sm,Geweke = ACC[sub_fid,:]

toc = time.perf_counter()    
print(f"Running time (20 sites): {toc-tic:0.4f} seconds")






