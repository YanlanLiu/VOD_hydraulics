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
from scipy.stats import norm

tic = time.perf_counter()

# =========================== control pannel =============================

parentpath = '/scratch/users/yanlan/'

versionpath = parentpath + 'Global_0817/'
inpath = parentpath+ 'Input_Global/'
pixelpath = versionpath+'Pixels/Input/'

MODE = 'VOD_SM_ET'
varnames, bounds = get_var_bounds(MODE)
SiteInfo = pd.read_csv('pixels_good.csv')

for fid in range(len(SiteInfo)):
    sitename = str(SiteInfo['row'].values[fid])+'_'+str(SiteInfo['col'].values[fid])
    print(sitename)
    Forcings,VOD,SOILM,ET,dLAI,discard_vod,discard_et,idx = readCLM(inpath,sitename)
    fname = 'Input_'+sitename+'.pkl'
    with open(pixelpath+fname,'wb') as f: pickle.dump((Forcings,VOD,SOILM,ET,dLAI,discard_vod,discard_et,idx),f)

