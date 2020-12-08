#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 16:45:02 2020

@author: yanlanliu
"""
from Utilities import LatLon

import numpy as np
import pandas as pd
from netCDF4 import Dataset    # Note: python is case-sensitive!
import numpy as np
import matplotlib.pyplot as plt


lat0 = np.arange(-60,90,0.25)+0.25/2
lon0 = np.arange(-180,180,0.25)+0.25/2
r = np.arange(len(lat0)); c = np.arange(len(lon0))
ID_C, ID_R  =np.meshgrid(c,r)
df0 = pd.DataFrame({'row':ID_R.flatten(), 'col':ID_C.flatten()},index=np.arange(len(ID_R.flatten())))

df_para = pd.read_csv('PARA.csv').dropna()
df_m = pd.merge(df0, df_para, how='left', on=['row', 'col'])
df_m['lat'], df_m['lon'] = LatLon(df_m['row'],df_m['col'])


df_para = pd.read_csv('STD_new.csv').dropna()
df_std = pd.merge(df0, df_para, how='left', on=['row', 'col'])
df_std['lat'], df_std['lon'] = LatLon(df_std['row'],df_std['col'])

# test = df_m.pivot(index='lat', columns='lon', values='g1').values
# plt.pcolormesh(lon0,lat0,test)

#%%
def write_nc(fname,vnames,units,standard_names,var_values):
    ncfile = Dataset(fname,mode='w',format='NETCDF4_CLASSIC') 

    lat_dim = ncfile.createDimension('lat', len(lat0))     # latitude axis
    lon_dim = ncfile.createDimension('lon', len(lon0))    # longitude axis
    
    lat = ncfile.createVariable('lat', np.float32, ('lat',))
    lat.units = 'degrees_north'
    lat.long_name = 'latitude'
    lat[:] = lat0
    
    
    lon = ncfile.createVariable('lon', np.float32, ('lon',))
    lon.units = 'degrees_east'
    lon.long_name = 'longitude'
    lon[:] = lon0
    
    for i,v in enumerate(vnames):
        g1 = ncfile.createVariable(v,np.float32,('lat','lon')) # note: unlimited dimension is leftmost
        g1.units = units[i]
        g1.standard_name = standard_names[i] # this is a CF standard name
        g1[:,:] = var_values[i]


fname = 'MDF_g1.nc'
vnames = ['g1'+itm for itm in ['_m','_std']]
units = ['kPa^{1/2}' for i in range(2)]
standard_names = ['Ensemble mean of g1 (the slope parameter in Medlyn\'s stomatal conductance model)',
                  'Standard deviation of g1 across ensembles']
var_values = [df_m.pivot(index='lat', columns='lon', values='g1').values,
              df_std.pivot(index='lat', columns='lon', values='g1').values]
write_nc(fname,vnames,units,standard_names,var_values)

#%%
fname = 'MDF_P50.nc'
vnames = ['P50'+itm for itm in ['_m','_std']]
units = ['MPa' for i in range(2)]
standard_names = ['Ensemble mean of P50 (the leaf water potential at 50% of xylem conductance)',
                  'Standard deviation of P50 across ensembles']
var_values = [df_m.pivot(index='lat', columns='lon', values='psi50X').values,
              df_std.pivot(index='lat', columns='lon', values='psi50X').values]
write_nc(fname,vnames,units,standard_names,var_values)

#%%
fname = 'MDF_gpmax.nc'
vnames = ['gpmax'+itm for itm in ['_m','_std']]
units = ['mm/hr/MPa' for i in range(2)]
standard_names = ['Ensemble mean of gpmax (maximum xylem conductance)',
                  'Standard deviation of gpmax across ensembles']
var_values = [df_m.pivot(index='lat', columns='lon', values='gpmax').values,
              df_std.pivot(index='lat', columns='lon', values='gpmax').values]
write_nc(fname,vnames,units,standard_names,var_values)

#%%
fname = 'MDF_C.nc'
vnames = ['C'+itm for itm in ['_m','_std']]
units = ['mm/MPa' for i in range(2)]
standard_names = ['Ensemble mean of C (vegetation capacitance)',
                  'Standard deviation of C across ensembles']
var_values = [df_m.pivot(index='lat', columns='lon', values='C').values,
              df_std.pivot(index='lat', columns='lon', values='C').values]
write_nc(fname,vnames,units,standard_names,var_values)


#%%
fname = 'MDF_P50s_P50x.nc'
vnames = ['P50s_P50x'+itm for itm in ['_m','_std']]
units = ['unitless' for i in range(2)]
standard_names = ['Ensemble mean of P50s/P50x (the ratio between leaf water potential at 50% of stomatal conductance and that at 50% of xylem conductance)',
                  'Standard deviation of P50s/P50x across ensembles']
var_values = [df_m.pivot(index='lat', columns='lon', values='lpx').values/df_m.pivot(index='lat', columns='lon', values='psi50X'),
              df_std.pivot(index='lat', columns='lon', values='lpx').values]
write_nc(fname,vnames,units,standard_names,var_values)

#%%


# ncfile = Dataset('traits_mean.nc',mode='w',format='NETCDF4_CLASSIC') 
# print(ncfile)

# lat_dim = ncfile.createDimension('lat', len(lat0))     # latitude axis
# lon_dim = ncfile.createDimension('lon', len(lon0))    # longitude axis

# lat = ncfile.createVariable('lat', np.float32, ('lat',))
# lat.units = 'degrees_north'
# lat.long_name = 'latitude'
# lat[:] = lat0


# lon = ncfile.createVariable('lon', np.float32, ('lon',))
# lon.units = 'degrees_east'
# lon.long_name = 'longitude'
# lon[:] = lon0

# g1 = ncfile.createVariable('g1_m',np.float32,('lat','lon')) # note: unlimited dimension is leftmost
# g1.units = 'kPa^{1/2}' # degrees Kelvin
# g1.standard_name = 'Ensemble mean of g1 (the slope parameter in Medlyn\'s stomatal conductance model)' # this is a CF standard name
# g1[:,:] = df_merged.pivot(index='lat', columns='lon', values='g1').values


# C = ncfile.createVariable('C_m',np.float32,('lat','lon')) # note: unlimited dimension is leftmost
# C.units = 'mm/MPa' 
# C.standard_name = 'Ensemble mean of C (vegetation capacitance)' # this is a CF standard name
# C[:,:] = df_merged.pivot(index='lat', columns='lon', values='C').values


# p50s = ncfile.createVariable('p50s_m',np.float32,('lat','lon')) # note: unlimited dimension is leftmost
# p50s.units = 'MPa' 
# p50s.standard_name = 'Ensemble mean of P50s (leaf water potential at 50% of stomatal conductance)' # this is a CF standard name
# p50s[:,:] = df_merged.pivot(index='lat', columns='lon', values='lpx').values


# gpmax = ncfile.createVariable('gpmax_m',np.float32,('lat','lon')) # note: unlimited dimension is leftmost
# gpmax.units = 'mm/hr/MPa' 
# gpmax.standard_name = 'Ensemble mean of gpmax (maximum xylem conductance)' # this is a CF standard name
# gpmax[:,:] = df_merged.pivot(index='lat', columns='lon', values='gpmax').values


# p50x = ncfile.createVariable('p50x_m',np.float32,('lat','lon')) # note: unlimited dimension is leftmost
# p50x.units = 'MPa' 
# p50x.standard_name = 'Ensemble mean P50x (leaf water potential at 50% of xylem conductance)' # this is a CF standard name
# p50x[:,:] = df_merged.pivot(index='lat', columns='lon', values='psi50X').values

# #%%

# df_std = pd.read_csv('STD_new.csv').dropna()
# # df_mean = pd.read_csv('PARA.csv').dropna()
# # for v in ['g1','psi50X','lpx','g1']
# lat = np.arange(-60,90,0.25)+0.25/2
# lon = np.arange(-180,180,0.25)+0.25/2
# r = np.arange(len(lat)); c = np.arange(len(lon))
# ID_C, ID_R  =np.meshgrid(c,r)
# df0 = pd.DataFrame({'row':ID_R.flatten(), 'col':ID_C.flatten()},index=np.arange(len(ID_R.flatten())))
# df_merged = pd.merge(df0, df_std, how='left', on=['row', 'col'])
# df_merged['lat'], df_merged['lon'] = LatLon(df_merged['row'],df_merged['col'])

# test = df_merged.pivot(index='lat', columns='lon', values='g1').values; #test[test>np.quantile(test,.99)] = np.nan
# plt.pcolormesh(lon,lat,test)

# ncfile = Dataset('traits_std.nc',mode='w',format='NETCDF4_CLASSIC') 
# print(ncfile)


# lat_dim = ncfile.createDimension('lat', len(lat0))     # latitude axis
# lon_dim = ncfile.createDimension('lon', len(lon0))    # longitude axis

# lat = ncfile.createVariable('lat', np.float32, ('lat',))
# lat.units = 'degrees_north'
# lat.long_name = 'latitude'
# lat[:] = lat0


# lon = ncfile.createVariable('lon', np.float32, ('lon',))
# lon.units = 'degrees_east'
# lon.long_name = 'longitude'
# lon[:] = lon0


# g1 = ncfile.createVariable('g1_std',np.float32,('lat','lon')) # note: unlimited dimension is leftmost
# g1.units = 'kPa^{1/2}' 
# g1.standard_name = 'Standard deviation across ensembles of g1 (the slope parameter in Medlyn\'s stomatal conductance model)' # this is a CF standard name
# tmp = df_merged.pivot(index='lat', columns='lon', values='g1').values; tmp[tmp>np.quantile(tmp,.99)] = np.nan
# g1[:,:] = tmp.copy()

# C = ncfile.createVariable('C_std',np.float32,('lat','lon')) # note: unlimited dimension is leftmost
# C.units = 'mm/MPa' 
# C.standard_name = 'Standard deviation across ensembles of C (vegetation capacitance)' # this is a CF standard name
# C[:,:] = df_merged.pivot(index='lat', columns='lon', values='C').values


# p50s = ncfile.createVariable('p50s_p50x_std',np.float32,('lat','lon')) # note: unlimited dimension is leftmost
# p50s.units = 'unitless' 
# p50s.standard_name = 'Standard deviation across ensembles of P50s/P50x (the ratio between leaf water potential at 50% of stomatal conductance and that at 50% of xylem conductance)' # this is a CF standard name
# p50s[:,:] = df_merged.pivot(index='lat', columns='lon', values='lpx').values

# gpmax = ncfile.createVariable('gpmax_std',np.float32,('lat','lon')) # note: unlimited dimension is leftmost
# gpmax.units = 'mm/hr/MPa' 
# gpmax.standard_name = 'Standard deviation across ensembles of gpmax (maximum xylem conductance)' # this is a CF standard name
# gpmax[:,:] = df_merged.pivot(index='lat', columns='lon', values='gpmax').values


# p50x = ncfile.createVariable('p50x_std',np.float32,('lat','lon')) # note: unlimited dimension is leftmost
# p50x.units = 'MPa' 
# p50x.standard_name = 'Standard deviation across ensembles of P50x (leaf water potential at 50% of xylem conductance)' # this is a CF standard name
# p50x[:,:] = df_merged.pivot(index='lat', columns='lon', values='psi50X').values




#%% Check nc
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt

f = Dataset('MDF_g1.nc',mode='r') 
lat = np.array(f['lat'][:])
lon = np.array(f['lon'][:])
print(f.variables)
plt.figure(figsize=(14.4,6))
plt.pcolormesh(lon,lat,f.variables['g1_m']);plt.colorbar()
