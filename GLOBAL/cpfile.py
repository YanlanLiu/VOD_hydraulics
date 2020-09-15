from shutil import copyfile
import pandas as pd
versionpath = '/scratch/users/yanlan/Global_0817/'
forwardpath = versionpath+'Forward/'
dstpath = versionpath+'Pixels/Forward'

pixels = pd.read_csv('pixels.csv')
for fid in range(len(pixels)):
    sitename = str(pixels['row'].values[fid])+'_'+str(pixels['col'].values[fid])
    fname = 'TS_VOD_SM_ET_'+sitename+'.pkl'
    copyfile(forwardpath+fname,dstpath+fname)
#copyfile(src, dst)
