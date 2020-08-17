import numpy as np
import pandas as pd
import os

arrayid = int(os.environ['SLURM_ARRAY_TASK_ID'])
#arrayid = 0

parentpath = '/scratch/users/yanlan/Input_Global/'

SiteInfo = pd.read_csv('SiteInfo_globe.csv').iloc[arrayid*1000:(arrayid+1)*1000].reset_index().drop(columns=['index','Unnamed: 0'])
#print(SiteInfo)
rlist = np.array(SiteInfo['row']); clist = np.array(SiteInfo['col'])

frac = []
for i in range(len(rlist)):
    print(i)
    sitename = str(rlist[i])+'_'+str(clist[i])+'.csv'
    ET = pd.read_csv(parentpath+'ALEXI/ET_'+sitename)
    VOD = pd.read_csv(parentpath+'AMSRE/VOD_'+sitename)
    GLDAS = pd.read_csv(parentpath+'Climate/GLDAS_'+sitename)
    if os.path.isfile(parentpath+'LAI/LAI_'+sitename):
        LAI = pd.read_csv(parentpath+'LAI/LAI_'+sitename)
        frac_lai = sum(~np.isnan(LAI['Lai']))/len(LAI) if 'Lai' in list(LAI) else 0
    else:
        frac_lai = np.nan
    frac.append([sum(~np.isnan(ET['ET']))/len(ET),sum(~np.isnan(VOD['VOD_am']))/len(VOD),sum(~np.isnan(GLDAS['Swnet_tavg']))/len(GLDAS),frac_lai])
    #print(list(ET),list(VOD),list(GLDAS),list(LAI))

frac = np.array(frac)
#print(frac)
df_frac = pd.DataFrame(frac,columns=['f_ET','f_VOD','f_CLM','f_LAI'])
#print(df_frac)

df_frac.to_csv(parentpath+'STATS/Nobs_'+str(arrayid).zfill(3)+'.csv')
