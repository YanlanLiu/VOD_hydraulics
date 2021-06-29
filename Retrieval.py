#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 22:11:33 2020

@author: yanlan
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import genextreme as gev
import warnings; warnings.simplefilter("ignore")
from myfun import readCLM, fitVOD_RMSE, AMIS
from myfun import get_var_bounds,dt, hour2day, hour2week
from myfun import OB,CONST,CLAPP,ca
from Utilities import MovAvg
import sys

# =========================== control pannel =============================
parentpath = '/scratch/users/yanlan/'
baseid = int(sys.argv[1])
arrayid = int(os.environ['SLURM_ARRAY_TASK_ID'])*94+baseid # 0-93
samplenum = (25,2000)
versionpath = parentpath + 'Global_0817/'; hyperpara = (0.1,0.05,20)

inpath = parentpath+'Input_Global/'
outpath = versionpath+'Output/'
MODE = 'VOD_SM_ET'

fid = int(arrayid)
chainid = 0

SiteInfo = pd.read_csv('SiteInfo.csv')
if fid>=len(SiteInfo):
     print("fid>len(SiteInfo)")
     sys.exit(1)
sitename = str(SiteInfo['row'][fid])+'_'+str(SiteInfo['col'][fid])
PREFIX = outpath+MODE+'_'+sitename+'_'+str(chainid).zfill(2)
print(PREFIX)

if SiteInfo.iloc[fid]['IGBP']==1:  # ENF
    def f_p50_prior(p50): return np.log(gev.pdf(-p50, 0.65, -4.43, 1.94)+1e-20)
    p50_init = 4.43
elif SiteInfo.iloc[fid]['IGBP']==2:  # EBF
    def f_p50_prior(p50): return np.log(gev.pdf(-p50, 1.08, -2.86, 2.92)+1e-20)
    p50_init = 2.86
elif (SiteInfo.iloc[fid]['IGBP']>5) and (SiteInfo.iloc[fid]['IGBP']<10):  # SHB and SAV
    def f_p50_prior(p50): return np.log(gev.pdf(-p50, 0.76, -3.64, 2.55)+1e-20)
    p50_init = 3.64
elif SiteInfo.iloc[fid]['IGBP']>9: #GRA and others
    def f_p50_prior(p50): return np.log(gev.pdf(-p50, 0.77, -1.86, 1.25)+1e-20)
    p50_init = 1.86
else: #DBF and MF
    def f_p50_prior(p50): return np.log(gev.pdf(-p50, 0.71, -2.23, 1.49)+1e-20)
    p50_init = 2.23
    
#%% =========================== read input =================================
Forcings,VOD,SOILM,ET,dLAI,discard_vod,discard_et,idx = readCLM(inpath,sitename)

VOD_ma = np.reshape(VOD,[-1,2])
VOD_ma = np.reshape(np.column_stack([MovAvg(VOD_ma[:,0],4),MovAvg(VOD_ma[:,1],4)]),[-1,])

Z_r,tx = (SiteInfo['Root depth'][fid]*1000,int(SiteInfo['Soil texture'][fid]))

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

#%% Calculations not affected by MCMC paramteres
RNET,TEMP,P,VPD,Psurf,GA,LAI,VegK = Forcings

N = len(RNET)
# Terms in Farquhar's model of biochemical demand for CO2
PAR = RNET/(CONST.Ephoton*CONST.NA)*1e6
T_C = TEMP-CONST.U3 # degree C
Kc = 300*np.exp(0.074*(T_C-25)) # umol/mol
Ko = 300*np.exp(0.015*(T_C-25)) # mmol/mol
cp = 36.9+1.18*(T_C-25)+0.036*(T_C-25)**2
Vcmax25 = SiteInfo['Vcmax25'][fid]
Vcmax0 = Vcmax25*np.exp(50*(TEMP-298)/(298*CONST.R*TEMP)) 
Jmax = Vcmax0*np.exp(1)
J = (OB.kai2*PAR+Jmax-np.sqrt((OB.kai2*PAR+Jmax)**2-4*OB.kai1*OB.kai2*PAR*Jmax))/2/OB.kai1

# Terms in Penman-Monteith Equation
VPD_kPa = VPD*Psurf
sV = 0.04145*np.exp(0.06088*T_C) #in Kpa/K
RNg = np.array(RNET*np.exp(-LAI*VegK)) # RNg is the net radiation on the ground surface; RNET-RNg is the net radiation on the leaf surface
petVnum = (sV*(RNET-RNg)+1.225*1000*VPD_kPa*GA)*(RNET>0)/CONST.lambda0*60*60  #kg/s/m2/CONST.lambda0*60*60 -> kPa/K times mm/hr
petVnumB = 1.26*(sV*RNg)/(sV+CONST.gammaV)/CONST.lambda0*60*60 # mm/hr

#%%
def advance_linearize(s2,phiL,ti,gpmax,C,psi50X,bexp,timestep):
    a = -1/(2*psi50X)
    phiS2 = phi0*(s2/n)**(-bexp)
    delta_phi = phiS2 - phiL
    
    f_const = gpmax*(1+a*phiL)*delta_phi
    f_x = gpmax*(a*delta_phi + (1+a*phiL)*(-1))
    f_y = gpmax*(1+a*phiL)*(phiS2*(-bexp)/s2) 
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

    p3 = phi0_mm*(sbot/n)**(-bexp)+m3 
    k3 = ksoil*(sbot/n)**(2*bexp) 
        
    phil_list = np.zeros([N,])
    et_list = np.zeros([N,])
    
    s1 = np.copy(sinit)
    s2 = np.copy(sinit) 
    phiL = phi0*(s2/n)**(-bexp) - 0.01
    
    s1_list = np.zeros([N,]); 

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
            tlist = np.zeros(tdiv)
            for subt in np.arange(tdiv):
                condS = max(min(1-phiL/(2*psi50L),1),0)
                tlist[subt] = get_ti(clm,condS)               
                s2, phiL = advance_linearize(s2,phiL,tlist[subt],gpmax,C,psi50X,bexp,dt/tdiv)
            ti = np.mean(tlist)
        
        ei= petVnumB[i]*(s1/n) 
        s1 = min(s1+(P[i]-ei)*dt/d1,n)
        
              
        p1 = phi0_mm*(s1/n)**(-bexp) + m1
        p2 = phi0_mm*(s2/n)**(-bexp) + m2
        k1 = ksoil*(s1/n)**(2*bexp)
        k2 = ksoil*(s2/n)**(2*bexp)
        f12 = 2/(1/k1+1/k2) * (p1-p2) / (m1-m2)*dt
        f23 = 2/(1/k2+1/k3) * (p2-p3) / (m2-m3)*dt
        s1 = max(s1-f12/d1,0.05)
        s2 = min(max(s2+f12/d2 - f23/d2,0.05),n)  
        phiL = max(psi50X*2,phiL)

        et_list[i] = ei+ti
        s1_list[i] = np.copy(s1)
    s1_list[np.isnan(s1_list)] = np.nanmean(s1_list); s1_list[s1_list>1] = 1; s1_list[s1_list<0] = 0
    
    return phil_list,et_list,s1_list

#%%
# ========================== MCMC sampling ============================== 
varnames, bounds = get_var_bounds(MODE)
scale = bounds[2]
idx_sigma_vod = varnames.index('sigma_vod')
valid_vod = ~np.isnan(VOD_ma); VOD_ma_valid = VOD_ma[valid_vod]

idx_sigma_et = varnames.index('sigma_et')
valid_et = ~np.isnan(ET); ET_valid = ET[valid_et]

idx_sigma_sm = varnames.index('sigma_sm')
valid_sm = ~np.isnan(SOILM); SOILM_valid = SOILM[valid_sm]
bins = np.arange(0,1.02,0.01)
counts, bin_edges = np.histogram(SOILM_valid, bins=bins, normed=True)
cdf1 = np.cumsum(counts)/sum(counts)

Nobs = sum(valid_vod)+sum(valid_et)+sum(valid_sm)
    

def Gaussian_loglik(theta0):
    theta = theta0*scale
    PSIL_hat,ET_hat,SM_hat = runhh_2soil_hydro(theta)
    ET_hat = hour2week(ET_hat,UNIT=24)[~discard_et][valid_et] # mm/hr -> mm/day
    dPSIL = hour2day(PSIL_hat,idx)[~discard_vod]
    VOD_hat = fitVOD_RMSE(dPSIL,dLAI,VOD_ma)[valid_vod]
    SM_hat = hour2day(SM_hat,idx)[~discard_vod][::2][valid_sm]
    
    sigma_VOD, sigma_ET,sigma_SM = (theta[idx_sigma_vod], theta[idx_sigma_et],theta[idx_sigma_sm])
    loglik_vod = np.nanmean(norm.logpdf(VOD_ma_valid,VOD_hat,sigma_VOD))
    loglik_et = np.nanmean(norm.logpdf(ET_valid,ET_hat,sigma_ET))
    
    if np.isfinite(np.nansum(SM_hat)) and np.nansum(SM_hat)>0:
        counts, bin_edges = np.histogram(SM_hat, bins=bins, normed=True)
        cdf2 = np.cumsum(counts)/sum(counts)
        SM_matched = np.array([bin_edges[np.abs(cdf1-cdf2[int(itm*100)]).argmin()] for itm in SM_hat])
        loglik_sm = np.nanmean(norm.logpdf(SOILM_valid,SM_matched,sigma_SM))
    else:
        loglik_sm = np.nan
    return (loglik_vod+loglik_et+loglik_sm)/3*Nobs+f_p50_prior(theta[2])
    

AMIS(Gaussian_loglik,PREFIX,varnames, bounds,p50_init,samplenum,hyperpara)


