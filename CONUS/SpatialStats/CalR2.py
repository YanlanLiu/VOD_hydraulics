import os
import numpy as np
import pandas as pd
import glob
import sys; sys.path.append("../Utilities/")
from newfun import readCLM
from Utilities import MovAvg
import pickle
import time


parentpath = '/scratch/users/yanlan/' 
#arrayid = int(os.environ['SLURM_ARRAY_TASK_ID']) # 0-14
#nsites_per_id = 1000


#parentpath = '/Volumes/ELEMENTS/VOD_hydraulics/'
arrayid = 83
nsites_per_id = 1

versionpath = parentpath + 'Retrieval_VOD_ET/'
inpath = parentpath+'Input/'
outpath = versionpath+'Output/'
#forwardpath = parentpath+'Forward_test/'
#r2path = parentpath+'R2_test/'
forwardpath = versionpath+'Forward/'
r2path = versionpath+'R2/'

SiteInfo = pd.read_csv('../Utilities/SiteInfo_US_full.csv')
MODE = 'VOD_ET_'

MissingList = []
R2 = np.zeros([nsites_per_id,2])+np.nan
RMSE = np.zeros([nsites_per_id,2])+np.nan
corr = np.zeros([nsites_per_id,2])+np.nan

def nancorr(yhat,y):
    tmpfilter = ~np.isnan(yhat+y)
    return np.corrcoef(yhat[tmpfilter],y[tmpfilter])[0,1]
    
def calmaxR2(SVOD,VOD):
    SST = np.nanmean((VOD-np.nanmean(VOD))**2)
    RMSE = np.sqrt(min(np.nanmean((VOD-SVOD)**2,axis=1)))
    R2 = 1-RMSE**2/SST
    R2_m = 1-np.nanmean((VOD-np.nanmean(SVOD,axis=0))**2)/SST
    corr = nancorr(np.nanmean(SVOD,axis=0),VOD)
    return max(R2,R2_m),RMSE,corr
#%%
tic = time.perf_counter()
for i,fid in enumerate(range(arrayid*nsites_per_id,(arrayid+1)*nsites_per_id)):
    sitename = str(SiteInfo['row'][fid])+'_'+str(SiteInfo['col'][fid])
    Forcings,VOD,ET,dLAI,discard_vod,discard_et,idx = readCLM(inpath,sitename)
    
    VOD_ma = np.reshape(VOD,[-1,2])
    VOD_ma = np.reshape(np.column_stack([MovAvg(VOD_ma[:,0],4),MovAvg(VOD_ma[:,1],4)]),[-1,])
    
    forwardname = forwardpath+MODE+sitename+'.pkl'
    if os.path.isfile(forwardname):
        with open(forwardname,'rb') as f:  # Python 3: open(..., 'rb')
            SVOD, SET, SPSIL,SPOPT = pickle.load(f)
        R2[i,0],RMSE[i,0],corr[i,0] = calmaxR2(SVOD,VOD_ma)
        R2[i,1],RMSE[i,1],corr[i,1] = calmaxR2(SET,ET)
        # R2[i,:] = np.array([calmaxR2(SVOD,VOD_ma),calmaxR2(SET,ET)])
        #R2[i,:] = np.array([calmaxR2(SVOD,VOD),calmaxR2(SET,ET)])
    else:
        MissingList.append(fid)
        
toc = time.perf_counter()
print(f"Running time (1000 sites): {toc-tic:0.4f} seconds")

#%%
r2name = r2path+'R2_'+str(arrayid)+'_1E3.pkl'
with open(r2name, 'wb') as f: 
    pickle.dump([R2, RMSE,corr,MissingList], f)
