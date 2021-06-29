#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 09:54:59 2020

@author: yanlan
"""

import numpy as np
import time
from scipy.stats import  multivariate_normal,bernoulli,uniform
from scipy import optimize
from Utilities import LatLon, toTimestamp,savitzky_golay, dailyAvg, dailyMin, rm_outlier
from datetime import datetime, timedelta
import pandas as pd
import pickle
import os
from copy import copy

UNIT_0 = 18e-6 # mol H2O/m2/s -> m/s H2O
UNIT_1 = 1.6*UNIT_0 # mol CO2 /m2/s -> m/s, H2O
UNIT_2 = 1e6 # Pa -> MPa
UNIT_3 = 273.15 # Degree C -> K
UNIT_4 = UNIT_0*3600*24 # mol H2O /m2/s -> mm/day,s H2O
ca = 400 # ppm, atmospheric CO2 concentration

nobsinaday = 8; dt = 24/nobsinaday; warmup = 182
# yrange = range(2003,2006)
start_date = datetime(2003,7,2); end_date = datetime(2006,1,1)
# t.strftime('%m/%d/%Y')
#yrange = range(2003,2012)
def readCLM(inpath,sitename):
    r,c = sitename.split('_')
    lat,lon= LatLon(int(r),int(c))
    df = pd.read_csv(inpath+'Climate/GLDAS_'+sitename+'.csv').drop(columns=['Unnamed: 0'])
    
    # local time, adjust Greenwich time to local time based on longitutde
    tt_gldas = np.array([datetime(2002,12,31,0,0)+timedelta(hours=3*i+round(lon/15)) for i in range(len(df))])
    idx1 = np.where(tt_gldas>start_date)[0][0]
    idx2 = np.where(tt_gldas<=end_date)[0][-1]+1
    ndays = int((idx2-idx1)/nobsinaday)
    df = df.iloc[idx1:idx2].reset_index().drop(columns=['index'])
    tt_gldas = tt_gldas[idx1:idx2]
    
    pmidx = np.argmin(np.abs([itm.timetuple().tm_hour-13.5 for itm in tt_gldas[:nobsinaday]]))
    amidx = np.argmin(np.abs([itm.timetuple().tm_hour-1.5 for itm in tt_gldas[:nobsinaday]]))
    
    TEMP = np.array(df['Tair_f_inst'])
    RNET = np.array(df['Swnet_tavg']); RNET[RNET<0] = 0
    P = np.array(df['Rainf_f_tavg'])*60*60 # mm/hr
    Psurf = np.array(df['Psurf_f_inst'])/1e3 # kPa
    VPD = T2ES(TEMP)/Psurf-np.array(df['Qair_f_inst'])
    Cpmol = 1005*28.97*1e-3 # J/kg/K*kg/mol -> J/mol/K
    GA = np.array(df['Qh_tavg'])/Cpmol/(np.array(df['AvgSurfT_inst'])-np.array(df['Tair_f_inst']))*0.02405  #mol/m2/s to m/s
    GA[GA<1e-6] = 1e-6; GA[GA>2] = 2 

    df_LAI = pd.read_csv(inpath+'LAI/LAI_'+sitename+'.csv')
    tt_lai = np.array([datetime.strptime(tmp,'%Y_%m_%d') for tmp in np.array(df_LAI['system:index'])])
    LAI = np.array(df_LAI['Lai'].interpolate(method='linear'))/10
    LAI = np.interp(toTimestamp(tt_gldas),toTimestamp(tt_lai),savitzky_golay(LAI,30,1))
    
    
    DOY = np.array([itm.timetuple().tm_yday+itm.timetuple().tm_hour/24 for itm in tt_gldas])
    leaf_angle_distr = 1
    VegK = LightExtinction(DOY,lat,leaf_angle_distr)
    
    discard = (dailyAvg(P,nobsinaday)>10/nobsinaday)+(dailyMin(TEMP-UNIT_3,nobsinaday)<-1)
    discard[:warmup] = True
    discard_vod = np.repeat(discard,2)
    discard_psil = np.repeat(discard,nobsinaday)
    discard_et = (hour2week(discard_psil,UNIT=1)>0.5)
    dLAI = np.repeat(dailyAvg(LAI,nobsinaday),2)[~discard_vod]

    amsr = pd.read_csv(inpath+'AMSRE/VOD_'+sitename+'.csv').drop(columns=['Unnamed: 0'])
    tt_amsr = np.arange(np.datetime64('2003-01-01'),np.datetime64('2012-01-01'))
    idx_vod = np.where(tt_amsr==tt_gldas[0].date())[0][0]
    amsr = amsr[idx_vod:idx_vod+ndays]
    VOD = np.reshape(np.column_stack([rm_outlier(amsr['VOD_am']),rm_outlier(amsr['VOD_pm'])]),[-1,])[~discard_vod]
    SOILM = rm_outlier(amsr['SOILM_am'])[~discard_vod[::2]]/100
    
    alexi = pd.read_csv(inpath+'ALEXI/ET_'+sitename+'.csv')
    tt_alexi = np.array([itm for y in range(2003,2012) for itm in np.arange(np.datetime64(str(y)+'-01-08'),np.datetime64(str(y+1)+'-01-01'), np.timedelta64(7,'D'))])
    ET = np.array(alexi['ET'])
    
    idx_et1 = np.where(tt_alexi>tt_gldas[0].date())[0][0]
    idx_et2 = np.where(tt_alexi<=tt_gldas[-1].date())[0][-1]+1
    ET = ET[idx_et1:idx_et2][~discard_et]        
    Forcings = (RNET,TEMP,P,VPD,Psurf,GA,LAI,VegK)
    
    return Forcings,VOD,SOILM,ET,dLAI,discard_vod,discard_et,[amidx,pmidx]

def LightExtinction(DOY,lat,x):
    B = (DOY-81)*2*np.pi/365
    ET = 9.87*np.sin(2*B)-7.53*np.cos(B)-1.5*np.sin(B)
    DA = 23.45*np.sin((284+DOY)*2*np.pi/365)# Deviation angle
    LST = np.mod(DOY*24*60,24*60)
    AST = LST+ET
    h = (AST-12*60)/4 # hour angle
    alpha = np.arcsin((np.sin(np.pi/180*lat)*np.sin(np.pi/180*DA)+np.cos(np.pi/180*lat)*np.cos(np.pi/180.*DA)*np.cos(np.pi/180*h)))*180/np.pi # solar altitude
    zenith_angle = 90-alpha
    Vegk = np.sqrt(x**2+np.tan(zenith_angle/180*np.pi)**2)/(x+1.774*(1+1.182)**(-0.733)) # Campbell and Norman 1998
    return Vegk

T2ES  = lambda x: 0.6108*np.exp(17.27*(x-UNIT_3)/(x-UNIT_3+237.3))# saturated water pressure, kPa


#%%
class OptimalBioChem:
    def __init__(self):
        self.koptj = 155.76 #  umol/m2/s
        self.Haj = 43.79 # kJ/mol
        self.Hdj = 200; # kJ/mol
        self.Toptj = 32.19+UNIT_3 # K
        self.koptv = 174.33 # umol/m2/s
        self.Hav = 61.21 # kJ/mol
        self.Hdv = 200 # kJ/mol
        self.Toptv = 37.74+UNIT_3 # K
        self.Coa = 210 # mmol/mol
        self.kai1 = 0.9
        self.kai2 = 0.3

class Constants:
    def __init__(self):
        self.R = 8.31*1e-3 # Gas constant, kJ/mol/K
        self.NA = 6.02e23 # Avogadro's constant, /mol
        self.hc = 2e-25 # Planck constant times light speed, J*s times m/s
        self.wavelen = 500e-9 # wavelength of light, m
        self.Ephoton = self.hc/self.wavelen
        self.ca = 400 # Atmospheric CO2 concentration, ppm
        self.Cpmol = 1005*28.97*1e-3 # J/kg/K*kg/mol -> J/mol/K
        self.lambda0 = 2.26*10**6 # heat of vaporization of water, J/kg
        self.gammaV = 100*1005/(self.lambda0*0.622) #in kPa/K, constant in PM equation
        self.a0 = 1.6 # relative diffusivity of h2o to co2 through stomata
        self.U3 = 273.15

class ClappHornberger:
    def __init__(self):
        self.psat = [12.1,9.0,21.8,78.6,47.8,29.9,35.6,63.0,15.3,49,40.5]
        self.thetas = [0.395,0.410,0.435,0.485,0.451,0.420,0.477,0.476,0.426,0.492,0.482]
        self.ksat = [1.056,0.938,0.208,0.0432,0.0417,0.0378,0.0102,0.0147,0.0130,0.0062,0.0077]

ca = 400
mpa2mm = 10**6/9.8
OB = OptimalBioChem()
CONST = Constants()
CLAPP = ClappHornberger()


def hour2day(X,idx):
    nday = int(len(X)/nobsinaday)
    if len(idx)==2:
        tmpidx = np.reshape(np.column_stack([np.arange(nday)*nobsinaday+idx[0],np.arange(nday)*nobsinaday+idx[1]]),[-1,])
    else:
        tmpidx = np.arange(nday)*nobsinaday+idx[0]
    return X[tmpidx,]


def hour2week(ET,UNIT=UNIT_4):
    ndays = int(len(ET)/nobsinaday)
    weeklyfilter = np.ones([ndays,])
    DOY = np.array([(start_date+timedelta(i)).timetuple().tm_yday for i in range(ndays)])
    weeklyfilter[0] = 0; weeklyfilter[(DOY==1)+(DOY==366)] = 0
    weeklyfilter = np.repeat(weeklyfilter,nobsinaday)[:len(ET)]
    
    ET = dailyAvg(ET[weeklyfilter==1],nobsinaday*7)*UNIT # mmol/m2/s -> mm/day
    return ET

def calVOD(x,psil,lai):
    a,b,c = x
    return (1 + a*psil)*(b + c*lai)

def RMSE_VOD(x,psil,lai,vod):
    return np.nansum((vod-calVOD(x,psil,lai))**2)

para0 = [0.3,0.64,0.04]
bounds_vod = ((0,50),(0,20),(0,50))


def fitVOD_RMSE(PSIL,LAI,VOD,return_popt=False):
    try:        
        res = optimize.minimize(RMSE_VOD,para0,args = (PSIL,LAI,VOD), bounds = bounds_vod)
    except RuntimeError as err:
        res.x = np.zeros([len(para0),]);print(err)
    VOD_fitted = calVOD(res.x,PSIL,LAI)
    if return_popt:
        return VOD_fitted, res.x
    else:
        return VOD_fitted

#%% MCMC functions

def get_var_bounds(MODE):
    if MODE=='VOD_ET' or MODE=='VOD_ET_ISO':
        varnames = ['g1','lpx','psi50X','gpmax','C','bexp','bc','sigma_et','sigma_vod','loglik']
        lowbound = np.array([0,0,0,0,0,1.5,0,0,0]); upbound = np.array([10,1,12,10,23,10,1,2,0.2])
    elif MODE=='VOD_SM' or MODE=='VOD_SM_ISO':
        varnames = ['g1','lpx','psi50X','gpmax','C','bexp','bc','sigma_sm','sigma_vod','loglik']
        lowbound = np.array([0,0,0,0,0,1.5,0,0,0]); upbound = np.array([10,1,12,10,23,10,1,0.2,0.2])
    elif MODE=='VOD_SM_ET' or MODE=='VOD_SM_ET_ISO':
        varnames = ['g1','lpx','psi50X','gpmax','C','bexp','bc','sigma_et','sigma_sm','sigma_vod','loglik']
        lowbound = np.array([0,0,0,0,0,1.5,0,0,0,0]); upbound = np.array([10,1,12,10,23,10,1,2,0.2,0.2])
    scale = np.max(abs(np.column_stack([lowbound,upbound])),axis=1)
    bounds = (lowbound/scale, upbound/scale, scale)
    return varnames, bounds

def AMIS(lik_fun,PREFIX,varnames, bounds, p50_init, samplenum, hyperpara = (0.1,0.05,20)): # AMIS sampling without parallel tempering
    numchunck, niter = samplenum
    p = int(len(varnames)-1)
    lowbound, upbound,scale = bounds
    
    mu = np.mean(np.column_stack([bounds[0],bounds[1]]),axis=1); mu[2] = p50_init/scale[2]
    sigma = 0.5*np.identity(p)
    tail_para = (mu,1**2*np.identity(p),0.2) # mu0, sigma0, ll
    r, power, K = hyperpara # hyper parameters
    rn = r/(1/K)**power
    
    paraname = PREFIX+'_sample_para.pkl' 
    if os.path.isfile(paraname):
        with open(paraname,'rb') as f: sample_para,sample_para0 = pickle.load(f)
        mu,sigma,rn,ii,theta,logp1 = sample_para
        for chunckid in range(20):
            if os.path.isfile(PREFIX+'_'+str(chunckid).zfill(2)+'.pickle'):
                chunckid0 = np.copy(chunckid)+1
            else:
                continue
    else:
        theta = AMIS_proposal(mu,mu,sigma,tail_para,bounds)
        logp1 = lik_fun(theta) 
        if np.isnan(logp1):logp1=-99999
        ii = 0
        sample_para0 = (mu,sigma,rn,ii,theta,logp1) # for use of restart
        chunckid0 = 0
    
    print(logp1)
    sample = np.copy(theta).reshape((-1,p))
    lik = [np.copy(logp1)]
    
    for chunckid in range(chunckid0,numchunck): 
        outname = PREFIX+'_'+str(chunckid).zfill(2)+'.pickle' 
        acc = 0 # acceptance rate within each chunck
        for i in range(niter):
            ii = ii+1#
            acc = acc*i/(i+1)            
            theta_star = AMIS_proposal(theta,mu,sigma,tail_para,bounds)
    
            # Evalute likelihood
            logp2 = lik_fun(theta_star) # before tempering
            logq2,singular = AMIS_prop_loglik(theta_star,mu,sigma,tail_para)
            if np.isfinite(logp2) and singular==False:
                logq1,singular1 = AMIS_prop_loglik(theta,mu,sigma,tail_para)

                # Accept with calculated probability
                logA = (logp2-logp1)-(logq2-logq1)
                if np.log(uniform.rvs())<logA:
                    acc = acc+1/(i+1)
                    theta = np.copy(theta_star)
                    logp1 = np.copy(logp2)
            
            sample = np.row_stack([sample,theta])
            lik = np.concatenate([lik,[logp1]])
            
             # Update proposal distribution
            if np.mod(i,K)==0:
                rn = rn*((ii+1)/(ii+2))**power
                mu = mu+rn*np.mean(sample[-K:]-mu,axis=0)
                sigma = sigma+rn*(np.dot(np.transpose(sample[-K:]-mu),sample[-K:]-mu)/K-sigma)        

                    
        det = np.linalg.det(sigma)
        print(logp1,acc,det,ii,rn)
        
        if ((acc<0.03) and (det<1e-100)) or ((acc<0.05) and (det<1e-120)) or acc<0.005 or det<1e-150: mu,sigma,rn,ii,theta,logp1 = sample_para0; print("restart..."); #det<1e-100 or 
        sample_para = (mu,sigma,rn,ii,theta,logp1)
        if acc>0.15: sample_para0 = copy(sample_para)

        sdf = pd.DataFrame(np.column_stack([sample*scale,lik]),columns = varnames)
        sdf.to_pickle(outname)
        sample = sample[-1,:]
        lik = [lik[-1]]
        with open(PREFIX+'_sample_para.pkl', 'wb') as f:
            pickle.dump((sample_para,sample_para0),f)       
        
        
    
MAX_STEP_TIME = 10 # sec
def AMIS_proposal(theta,mu,sigma,tail_para,bounds):
    lowbound, upbound,scale = bounds
    mu0,sigma0,ll = tail_para
    startTime_for_tictoc = time.time()
    while True:
        if bernoulli.rvs(ll,size=1) ==1:
            theta_star = multivariate_normal.rvs(mu0,sigma0,size = 1)
        else:
            theta_star = multivariate_normal.rvs(mu,sigma,size = 1)
       
        outbound = len([i for i,im in enumerate(theta_star) if ((lowbound[i]>im) or (im>upbound[i]))])
        if (outbound==0): 
            break
        if (time.time() - startTime_for_tictoc)>MAX_STEP_TIME:
            theta_star = np.copy(theta)
            break  
    return theta_star

def AMIS_prop_loglik(theta,mu,sigma,tail_para):
    mu0,sigma0,ll = tail_para
    p2 = multivariate_normal.pdf(theta,mu,sigma,allow_singular=True)
    singular = False
    if p2==1:
        p2 = 0
        singular = True
    return np.log(multivariate_normal.pdf(theta,mu0,sigma0)*ll+p2*(1-ll)),singular


def GetTrace(flist0,warmup=0):
    flist = [itm for itm in flist0 if os.path.isfile(itm)]
    flist.sort()  
    chunck_idx = len(flist[0])-9
    chain_idx = len(flist[0])-11

    for outname in flist:
        trace = pd.read_pickle(outname)

        chainid = int(outname[chain_idx:chain_idx+1])
        if outname==flist[0]:varnames = list(trace)
        niter = len(trace)-1
        chunckid = int(outname[chunck_idx:chunck_idx+2])
        tmp = pd.DataFrame(data={'index':np.arange(0,niter),'chain':chainid,'chunk':chunckid})
        tmp['step'] = chunckid*niter+tmp['index']
        for para in varnames: tmp[para] = np.array(trace[para][1:])
        
        if outname==flist[0]: trace_df = tmp 
        else: trace_df = pd.concat([trace_df,tmp])
    trace_df = trace_df[trace_df['step']>trace_df['step'].max()*warmup].sort_values(['step']).dropna().reset_index().drop(columns=['index','level_0','chain','chunk','step'])
    return trace_df
