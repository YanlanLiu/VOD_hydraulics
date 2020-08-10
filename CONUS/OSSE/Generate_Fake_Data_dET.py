#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 20:21:37 2020

@author: yanlan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 11:19:06 2020

@author: yanlan
"""

from random import randint
import pickle
import os
import numpy as np
import pandas as pd
import warnings; warnings.simplefilter("ignore")
import sys; sys.path.append("../Utilities/")
from newfun import readCLM # newfun_full
from newfun import fitVOD_RMSE,calVOD,dt, hour2day, hour2week
from newfun import OB,CONST,CLAPP,ca
from newfun import GetTrace
from Utilities import MovAvg, nanOLS, dailyAvg
import time

tic = time.perf_counter()

# =========================== control pannel =============================

# parentpath = '/scratch/users/yanlan/'
# arrayid = int(os.environ['SLURM_ARRAY_TASK_ID']) # 0-119
# nsites_per_id = 1
# warmup, nsample,thinning = (0.8,200,50)

parentpath = '/Volumes/ELEMENTS/VOD_hydraulics/'
arrayid = 0
nsites_per_id = 53

inpath = parentpath+'Input/'
datapath = parentpath + 'OSSE_dET/FakeData/'
# datapath = parentpath + 'OSSE2/FakeData/'

SiteInfo = pd.read_csv('SiteInfo_reps_53.csv')
MODE = 'AM_PM_ET'
outpath = parentpath+'Retrieval_0705/Output/'
varnames = ['g1','lpx','psi50X','gpmax','C','bexp','bc','sigma_et','sigma_vod','loglik']


    

for fid in range(arrayid*nsites_per_id,(arrayid+1)*nsites_per_id):

    sitename = str(SiteInfo['row'][fid])+'_'+str(SiteInfo['col'][fid])
    PREFIX = outpath+MODE+'_'+sitename+'_'
    
    
    Forcings,VOD,SOILM,ET,dLAI,discard_vod,discard_et,idx = readCLM(inpath,sitename)
    
    VOD_ma = np.reshape(VOD,[-1,2])
    VOD_ma = np.reshape(np.column_stack([MovAvg(VOD_ma[:,0],4),MovAvg(VOD_ma[:,1],4)]),[-1,])
    print(fid,sum(~np.isnan(VOD)),sum(~np.isnan(ET)),sum(~np.isnan(SOILM)))
    
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
    
    # Calculations not affected by MCMC paramteres
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

   # Vcmax0 = OB.koptv*OB.Hdv*np.exp(OB.Hav*(TEMP-OB.Toptv)/TEMP/CONST.R/OB.Toptv)/(OB.Hdv-OB.Hav*(1-np.exp(OB.Hav*(TEMP-OB.Toptv)/TEMP/CONST.R/OB.Toptv)))
   # Jmax = OB.koptj*OB.Hdj*np.exp(OB.Haj*(TEMP-OB.Toptj)/TEMP/CONST.R/OB.Toptj)/(OB.Hdj-OB.Haj*(1-np.exp(OB.Haj*(TEMP-OB.Toptj)/TEMP/CONST.R/OB.Toptj))) 
    J = (OB.kai2*PAR+Jmax-np.sqrt((OB.kai2*PAR+Jmax)**2-4*OB.kai1*OB.kai2*PAR*Jmax))/2/OB.kai1
    
    # Terms in Penman-Monteith Equation
    VPD_kPa = VPD*101.325#VPD*Psurf
    sV = 0.04145*np.exp(0.06088*T_C) #in Kpa
    RNg = np.array(RNET*np.exp(-LAI*VegK))
    petVnum = (sV*(RNET-RNg)+1.225*1000*VPD_kPa*GA)*(RNET>0)/CONST.lambda0*60*60  #kg/s/m2/CONST.lambda0*60*60
    # (sV*(rnmg-1*RNgg) + 1.225*1000*myvpd*myga)*(myrn > 0)
    petVnumB = 1.26*(sV*RNg)/(sV+CONST.gammaV)/CONST.lambda0*60*60 
    

    def advance_linearize(s2,phiL,ti,gpmax,C,psi50X,bexp,timestep):
        a = -1/(2*psi50X)
        # f_const = gpmax*(1+a*phiL)*(phi0*(s2/n)**(-bexp) - phiL)
        # f_x = gpmax*((a)*(phi0*(s2/n)**(-bexp) - phiL) + (1+a*phiL)*( - 1))
        # f_y = gpmax*(1+a*phiL)*(phi0*n**bexp*s2**(-bexp-1)*(-bexp))
        phiS2 = phi0*(s2/n)**(-bexp)
        delta_phi = phiS2 - phiL
        
        f_const = gpmax*(1+a*phiL)*delta_phi
        f_x = gpmax*(a*delta_phi + (1+a*phiL)*(-1))
        f_y = gpmax*(1+a*phiL)*(phiS2*(-bexp)/s2-phiL) # need to double check
        # f_y = gpmax*(1+a*phiL)*(phiS2*(-bexp)/s2) # need to double check
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
        # et_list = np.zeros([N,])
        
        s1 = np.copy(sinit)
        s2 = np.copy(sinit) 
        phiL = phi0*(s2/n)**(-bexp) - 0.01
        
        s1_list = np.zeros([N,]); s2_list = np.zeros([N,])
        e_list = np.zeros([N,]); t_list = np.zeros([N,])
    
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
                    s2, phiL = advance_linearize(s2,phiL,ti,gpmax,C,psi50X,bexp,dt/tdiv)
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
            # et_list[i] = ei+ti
            
            s1_list[i] = np.copy(s1); s2_list[i] = np.copy(s2)
            e_list[i] = np.copy(ei); t_list[i] = np.copy(ti)

        s1_list[np.isnan(s1_list)] = np.nanmean(s1_list); s1_list[s1_list>1] = 1; s1_list[s1_list<0] = 0
        return phil_list,e_list+t_list,s1_list
    

    # trace = GetTrace(PREFIX,varnames,0,optimal=False)
    # trace = trace[trace['step']>trace['step'].max()*0.8].reset_index().drop(columns=['index'])
    # # print(trace.mean())
    # theta = np.array(trace[varnames[:-1]].mean())
    
    # if SiteInfo.iloc[fid]['IGBP']==1:
    #     p50_range = [2,12]
    # elif SiteInfo.iloc[fid]['IGBP']>9:
    #     p50_range = [0.3,4]
    # else:
    #     p50_range = [0.5,5]
    # if theta[2]<p50_range[0] or theta[2]>p50_range[1]:
    #     theta[2] = np.random.uniform(p50_range[0],p50_range[1])
    # print(SiteInfo.iloc[fid]['IGBP'],theta[2])
    
    with open(datapath+'Para_'+sitename+'.pkl', 'rb') as f: theta,popt = pickle.load(f)
    # print(popt)
    PSIL_hat,ET_hat,S1_hat = runhh_2soil_hydro(theta)
    
    dPSIL = hour2day(PSIL_hat,idx)[~discard_vod]
    VOD_hat,popt = fitVOD_RMSE(dPSIL,dLAI,VOD_ma,return_popt=True)
    if popt[0] == 0: 
        popt[0] = 0.01
        VOD_hat = calVOD(popt,dPSIL,dLAI)
    
    # print(popt)
    # with open(datapath+'Para_'+sitename+'.pkl', 'wb') as f: pickle.dump((theta,popt), f)
    
    # ET_hat = hour2week(ET_hat,UNIT=24)[~discard_et] # weekly ET, mm/day
    ET_hat = dailyAvg(ET_hat,8)[~discard_vod[::2]]*24 # daily ET, mm/day
    S1_hat = hour2day(S1_hat,idx)[~discard_vod][::2]
    
    
    VOD_hat[np.isnan(VOD_ma)] = np.nan
    # ET_hat[np.isnan(ET)] = np.nan
    S1_hat[np.isnan(SOILM)] = np.nan
   
    s_vod = [0.005,0.03,0.05]
    s_et = [0.05,0.3,0.5]
    s_sm = [0.005,0.03,0.05]
    for noise_level in [0,1,2]:
        VOD_fake = VOD_hat+np.random.normal(0,s_vod[noise_level],VOD.shape)
        ET_fake = ET_hat+np.random.normal(0,s_et[noise_level],ET_hat.shape)
        SOILM_fake = S1_hat+np.random.normal(0,s_sm[noise_level],SOILM.shape)
        SOILM_fake[SOILM_fake>1] = 1
        SOILM_fake[SOILM_fake<0] = 0
        # S1_hat[np.abs(SOILM_fake-np.nanmean(SOILM_fake))>2*np.nanstd(SOILM_fake)] = np.nan
        SOILM_fake[np.abs(SOILM_fake-np.nanmean(SOILM_fake))>2*np.nanstd(SOILM_fake)] = np.nan
        
        # SOILM_fake[IsOutlier(SOILM_fake)] = np.nan
        # plt.figure()
        # plt.plot(VOD_fake)
        # plt.plot(VOD_hat)
        # plt.figure()
        # plt.plot(ET_hat)
        # plt.plot(ET_fake)
        # plt.figure()
        # plt.plot(S1_hat)
        # plt.plot(SOILM_fake)

        with open(datapath+'Gen_'+sitename+'_'+str(noise_level)+'.pkl', 'wb') as f: pickle.dump((VOD_fake,ET_fake,SOILM_fake), f)
    



