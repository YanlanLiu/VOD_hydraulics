# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:06:44 2020

@author: yanlan
"""

# Find soil type
# Soil properties from 
# http://www.fao.org/soils-portal/soil-survey/soil-maps-and-databases/harmonized-world-soil-database-v12/en/
# topsoil (0-30 cm) and subsoil (30-100 cm). 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Utilities import LatLon_r,LatLon

#r,c = LatLon_r(	-37.4222,144.0944)
#LatLon(509,1296)
# rlist = [290,433,509]; clist = [1045,861,1296]
# IGBP = [4,4,2] # main land cover from MODIS
# SiteInfo = pd.DataFrame({'row':rlist,'col':clist})
traitpath = '/Volumes/ELEMENTS/VOD_hydraulics/TraitData/'
SiteInfo = pd.read_csv('SiteInfo_globe_short.csv').drop(columns=['Unnamed: 0'])
#%% Soil texture
#archive_path = 'F:/ReaserchArchive/201612_Mortality/'
#fname = archive_path+'GIS/Climate/ASCII/soil_rsp.txt'
fname = traitpath+'soil_rsp.txt'
with open(fname,'r') as f:
    content = f.readlines()
SID = np.loadtxt(fname,skiprows = 6)
dd = 180/SID.shape[0]

plt.imshow(SID);plt.colorbar()
#df = pd.read_excel(archive_path+'Data/FAO_Soil/HWSD_DATA.xlsx')
df = pd.read_excel(traitpath+'HWSD_DATA.xlsx')
#soiltexture  = ['T_SAND','T_SILT','T_CLAY','S_SAND','S_SILT','S_CLAY']
soiltexture  = ['T_SAND','T_SILT','T_CLAY']

PCT = list([])
for i in range(len(SiteInfo)):
    r = SiteInfo['row'][i]; c = SiteInfo['col'][i]
    lat,lon=LatLon_r(*LatLon(r,c),dd=0.05)
    pct_id = [list(df).index(vname) for vname in soiltexture]
    PCT.append(list(df.loc[SID[lat,lon]][pct_id]))
SiteInfo = pd.concat([SiteInfo,pd.DataFrame(PCT,columns=soiltexture)],axis=1)

#%%
def FindTexture(pct):
    sand,silt,clay = pct
    if silt+1.5*clay<15: 
        tx = 0 # sand
    elif silt+1.5*clay>=15 and silt+2*clay<30: 
        tx = 1 # loamy sand
    elif (7<=clay<20 and sand>52 and silt+2*clay>=30) or (clay<7 and silt<50 and silt+2*clay>=30):
        tx = 2 # sandy loam
    elif (silt>=50 and 12<=clay<27) or (50<=silt<80 and clay<12):
        tx = 3 # silt loam
    elif 7<=clay<27 and 28<=silt<50 and sand <=52:
        tx = 4 # loam
    elif 20<=clay<35 and silt<28 and sand >45:
        tx = 5 # sandy clay loam
    elif 27<=clay<40 and sand<=20:
        tx = 6 # silt clay loam
    elif 27<=clay<40 and 20<sand<=45:
        tx = 7 # clay loam
    elif clay>=35 and sand>45:
        tx = 8 # sandy clay
    elif clay>=40 and silt>=40:
        tx = 9 # silt clay
    elif clay>=40 and sand<=45 and silt<40:
        tx = 10 # clay  
    elif silt>=80 and clay<12: # silt
        tx = 3 # silt
    else:
        tx = 4
    return int(tx)

tx = np.zeros([len(SiteInfo),])-1
for i in range(len(SiteInfo)):
    tx[i] = FindTexture(np.array(SiteInfo[soiltexture].iloc[i]))
SiteInfo['Soil texture'] = tx
#%% Maximum rooting depth, need to solve
import netCDF4 as nc4
import glob
inpath = traitpath+"MaxRoot/"
# inpath = 'F:\Data\MaxRoot//'
RD = np.zeros([len(SiteInfo),])-1
# radius = (0.25/2)**2
# flist = glob.glob(inpath+'*.nc')
# # for fname in [flist[3]]:#flist[::-1]:
# fname = flist[3]
# print(fname)
# f = nc4.Dataset(fname)
# dd = 0.25/2
# RootDepth = np.array(f.variables['root_depth']); RootDepth[RootDepth<0] = np.nan
# Latitude = np.array(f.variables['lat'])
# Longitude = np.array(f.variables['lon'])
# XX,YY = np.meshgrid(Longitude,Latitude)

#%%
dd = 0.25/2
for fname in flist:
    print(fname)
    f = nc4.Dataset(fname)
    RootDepth = np.array(f.variables['root_depth']); RootDepth[RootDepth<0] = np.nan
    Latitude = np.array(f.variables['lat'])
    Longitude = np.array(f.variables['lon'])

    for i in  range(len(SiteInfo)):
        # if RD[i]==-1:
        # if np.mod(i,1000)==0:print(i)
        lat,lon=LatLon(SiteInfo['row'][i],SiteInfo['col'][i])
        lat_dd = lat+np.array([-dd,dd]); lon_dd = lon+np.array([-dd,dd])
        rr = np.where((Latitude>=lat_dd[0])*(Latitude<=lat_dd[1]))[0]
        cc = np.where((Longitude>=lon_dd[0])*(Longitude<=lon_dd[1]))[0]
        if len(rr)>0 and len(cc)>0:
            grid = RootDepth[rr.min():rr.max()+1,cc.min():cc.max()+1]
            if np.nanmean(grid)>0:
                if np.nanmedian(grid)<0.3:
                    RD[i] = np.nanmean(grid)
                else:
                    RD[i] = np.nanmedian(grid)
            # distance = (YY-lat)**2+(XX-lon)**2
            # tmpfilter = (distance<radius)
            # if np.sum(tmpfilter)>0: 
            #     tmpfilter1 = tmpfilter*(RootDepth<10)
            #     RD[i] = np.nanmean(RootDepth[tmpfilter1]) if np.sum(tmpfilter1)>0 else np.nanmedian(RootDepth[tmpfilter])

RD[RD==-1] = np.nan
RD[RD>10] = 10; RD[RD<0.1] = 0.1
plt.hist(RD)

SiteInfo['Root depth'] = RD


# %%
heatmap1_data = pd.pivot_table(SiteInfo, values='Root depth', index='row', columns='col')
plt.imshow(heatmap1_data);plt.colorbar()
plt.title('Root depth')

#%% Koeppen-Geiger climate classification
#KG = pd.read_csv('F:\Data\Koeppen-Geiger-ASCII\\Koeppen-Geiger-ASCII.csv')
KG = pd.read_csv(traitpath+'/Koeppen-Geiger-ASCII.csv')

KGtype = []
for i in range(len(SiteInfo)):
    lat,lon=LatLon(SiteInfo['row'][i],SiteInfo['col'][i])
    distance = (KG['Lat']-lat)**2+(KG['Lon']-lon)**2
    KGtype.append(KG['Cls'][distance.idxmin()])
    
SiteInfo['Climate type'] = KGtype

#%%
from netCDF4 import Dataset
f = Dataset(traitpath+'GLDASp4_domveg_025d.nc4')
domveg = np.flipud(f.variables['GLDAS_domveg'][0,:,:])#[160:260,200:470]  # US
plt.imshow(domveg)
row = np.array(SiteInfo['row']); col = np.array(SiteInfo['col'])
IGBP = np.array([domveg[row[i],col[i]] for i in range(len(SiteInfo))])
SiteInfo['IGBP'] = IGBP.astype('int')

#%% Root type (Jackson et al., 1996, Oecologia)
# Index following the order of Table 1, starting from 0
RootType = []
RootType = -np.ones([len(SiteInfo),])
RootType[IGBP==12] = 1 # Cropland, crops
RootType[IGBP==14] = 1 # Cropland/Natural Vegetation Mosaic, crops
# RootType[IGBP==16] = 2 # Barren or Sparsely Vegetated, desert
RootType[IGBP==6] = 3 # Closed Shrublands, Sclerophyllous shrubs
RootType[IGBP==7] = 3 # Open Shrublands, Sclerophyllous shrubs
RootType[IGBP==10] = 6 # Grassland, Temperate grassland
RootType[IGBP==8] = 9 # Woody Savannas, Tropical grasslandsavann
RootType[IGBP==9] = 9 # Savannas, Tropical grasslandsavann


for i, root in enumerate(RootType):
    if root == -1:
        climate = SiteInfo['Climate type'][i]
        igbp = SiteInfo['IGBP'][i]

        if climate.find('Dfc') or climate.find('Dfd') or climate.find('Dw') or climate.find('E'): # boreal
            if igbp>=1 and igbp<=5: # forest
                RootType[i] = 0 # boreal forest
                
        if climate.find('B') or climate.find('C') or climate.find('Dwa') or climate.find('Dwb'): # temperate
            if igbp==1 or igbp==3: # Evergreen Needleleaf Forest, Deciduous Needleleaf Forest 
                RootType[i] = 4 # Temperate coniferous forest
            elif igbp==2 or igbp==4 or igbp==5:# Evergreen Broadleaf Forest, Deciduous Broadleaf Forest, Mixed Forest  
                RootType[i] = 5 # Temperate deciduous forest
            
        if climate.find('A')>=0: # tropical
            if igbp==3 or igbp==4 or igbp==5: # Deciduous Needleleaf Forest, Deciduous Broadleaf Forest, Mixed Forest 
                RootType[i] = 7 # Tropical deciduous forest
            elif igbp==1 or igbp==2: # Evergreen Needleleaf Forest , Evergreen Broadleaf Forest
                RootType[i] = 8 # Tropical evergreen forest
        
        if climate.find('ET'): # tundra
            if igbp>=18: # Wooded Tundra, Mixed Tundra, Bare Ground Tundra 
                RootType[i] = 10 # Tundra

SiteInfo['Root type'] = RootType

#%%
# https://nph.onlinelibrary.wiley.com/action/downloadSupplement?doi=10.1111%2Fnph.14623&file=nph14623-sup-0003-NotesS1.pdf

# SiteInfo = pd.read_csv('SiteInfo_US_full.csv')

Vcmax25 = np.zeros([len(SiteInfo),])+np.nan
IGBP = np.array(SiteInfo['IGBP'])
for i in range(len(Vcmax25)):
    if IGBP[i] in [2]: # evergreen broadleaf
        Vcmax25[i] = 35.5
    elif IGBP[i] in [1,3]: # evergreen needleleaf and deciduous needleleaf
        Vcmax25[i] = 71.1
    elif IGBP[i] in [4,5]: # deciduous broadleaf 
        Vcmax25[i] = 55
    elif IGBP[i] in [10,14]: # C3 grass/crop
        Vcmax25[i] = 62.2
    else:
        Vcmax25[i] = 54
SiteInfo['Vcmax25'] = Vcmax25

# SiteInfo.to_csv('SiteInfo_US_full.csv')
SiteInfo0 = SiteInfo.copy()
#%%
tmp = SiteInfo.dropna().reset_index().drop(columns='index')
tmp.to_csv('SiteInfo_globe_full.csv')

#%%
# SiteInfo.to_csv('SiteInfo_US_full.csv')
    
#for i in range(len(SiteInfo)):
#    climate = SiteInfo['Climate type'][i]
#    pft = SiteInfo['IGBP'][i]
#    root = -1 # does not belong to any catogory
#    if pft.find('Shrublands')>=0:
#        root = 3
#    elif pft.find('Croplands')>=0:
#        root = 1
#    elif climate.find('A')>=0: # Tropical
#        if pft.find('Deciduous')>=0:
#            root = 7
#        elif pft.find('Evergreen')>=0:
#            root = 8
#        elif pft.find('Savannas')>=0 or pft.find('Grasslands')>=0:
#            root  = 9
#    elif climate.find('B')>=0: # Dessert
#        root = 2
#    elif climate.find('C')>=0: # Temperate
#        if pft.find('Needleleaf')>=0:
#            root = 4
#        elif pft.find('Deciduous')>=0:
#            root = 5
#        elif pft.find('Grasslands')>=0:
#            root  = 6
#    elif climate.find('D')>=0: # Boreal
#        root = 0
#    elif climate.find('E')>=0: # Tundra
#        root = 10
#    RootType.append(root)

   

#%%
#PCT = []
#for i in range(len(siteinfo)):
#    lat = siteinfo['Latitude'][i]; lon = siteinfo['Longitude'][i]
#    r = int(round((90-dd/2-lat)/dd))
#    c = int(round((lon-(-180+dd/2))/dd))
#    pct_id = [list(df).index(vname) for vname in soiltexture]
#    PCT.append(list(df.loc[SID[r,c]][pct_id]))
#PCTdf = pd.DataFrame(PCT,columns=soiltexture)
##%%
#PCTdf.mean(axis=1,skipna=True).isnull().sum()
#PCTdf.mean(axis=1,skipna=False).isnull().sum()
##pd.concat([siteinfo, PCTdf], axis=1, sort=False).to_csv('SiteInfo.csv')
##pct_id = [list(df).index(vname) for vname in ['T_SAND','T_SILT','T_CLAY']]
##pct = np.array(df.loc[SID[r,c]][pct_id])
##b0 = -3.14-0.00222*pct[2]**2-0.00003484*pct[0]**2*pct[2]
##n0 = 0.332-7.251e-4*pct[0]+0.1276*np.log10(pct[2])
##mPsi0 = -100*np.exp(-4.396-0.0715*pct[2]-0.0004880*pct[0]**2-0.00004285*pct[0]**2*pct[2]) # MPa
##K0 = 2.778e-6*np.exp(12.012-0.0755*pct[0]+(-3.895+0.03671*pct[0]-0.1103*pct[2]+8.7546e-4*pct[0]**2)) # m/s
#
##%% Koeppen-Geiger climate classification
#siteinfo = pd.read_csv('SiteInfo.csv')
#KG = pd.read_csv('I:\Data\Koeppen-Geiger-ASCII\\Koeppen-Geiger-ASCII.csv')
#KGtype = []
#for i in range(len(siteinfo)):
#    distance = (KG['Lat']-siteinfo['Latitude'][i])**2+(KG['Lon']-siteinfo['Longitude'][i])**2
#    KGtype.append(KG['Cls'][distance.idxmin()])
#siteinfo['Climate type'] = KGtype
##siteinfo.to_csv('SiteInfo.csv')
#
##%% Root type (Jackson et al., 1996, Oecologia), 
## Index following the order of Table 1, starting from 0
#RootType = []
#for i in range(len(siteinfo)):
#    climate = siteinfo['Climate type'][i]
#    pft = siteinfo['IGBP'][i]
#    root = -1 # does not belong to any catogory
#    if pft.find('Shrublands')>=0:
#        root = 3
#    elif pft.find('Croplands')>=0:
#        root = 1
#    elif climate.find('A')>=0: # Tropical
#        if pft.find('Deciduous')>=0:
#            root = 7
#        elif pft.find('Evergreen')>=0:
#            root = 8
#        elif pft.find('Savannas')>=0 or pft.find('Grasslands')>=0:
#            root  = 9
#    elif climate.find('B')>=0: # Dessert
#        root = 2
#    elif climate.find('C')>=0: # Temperate
#        if pft.find('Needleleaf')>=0:
#            root = 4
#        elif pft.find('Deciduous')>=0:
#            root = 5
#        elif pft.find('Grasslands')>=0:
#            root  = 6
#    elif climate.find('D')>=0: # Boreal
#        root = 0
#    elif climate.find('E')>=0: # Tundra
#        root = 10
#    RootType.append(root)
#
#siteinfo['Root type'] = RootType    
##siteinfo.to_csv('SiteInfo.csv')
##%%
#siteinfo = pd.read_csv('SiteInfo.csv') # rigorous dataset from batch processing
#selected = siteinfo.loc[(siteinfo['Data length']>=2) & (siteinfo['IGBP']!='WET (Permanent Wetlands)')]
#selected.to_csv('SiteInfo_Selected.csv')
