#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 20:25:44 2020

@author: yanlan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 19:52:46 2020

@author: yanlan
"""

import numpy as np
import pandas as pd
import os

arrayid = 0
# arrayid = int(os.environ['SLURM_ARRAY_TASK_ID'])
yrange = np.arange(2003,2004)
#datapath = 'D:/Data/VOD/AMSRE/'
SiteInfo = np.array(pd.read_csv('SiteInfo_US.csv').iloc[arrayid*100:(arrayid+1)*100])[:,1:]
rlist = SiteInfo[:,0]; clist = SiteInfo[:,1]

PREFIX = r"/Volumes/ELEMENTS/D/Data/ET/ALEXIET_annual/ET"
def readET_ts(PREFIX,rlist,clist):
    ET_ts = np.transpose(np.array([[] for i in range(len(rlist))]))
    # tt = np.array([],dtype='datetime64')
    for year in yrange:
        print(year)
        ET_yr = np.load(PREFIX+str(year)+'.npy')  
        tmp = np.transpose(np.array([ET_yr[rlist[i],clist[i],:] for i in range(len(rlist))]))
        ET_ts = np.concatenate([ET_ts,tmp],axis=0)
        # tt = np.concatenate([tt,np.arange(np.datetime64(str(year)+'-01-08'),np.datetime64(str(year+1)+'-01-01'), np.timedelta64(7,'D'))])
    return ET_ts#,tt

ET_ts = readET_ts(PREFIX,rlist,clist)

#%%
for i in range(len(rlist)):
    tmp_df = pd.DataFrame({'ET':ET_ts[:,i]})
    tmp_df.to_csv('../ALEXI/ET_'+str(rlist[i])+'_'+str(clist[i])+'.csv')