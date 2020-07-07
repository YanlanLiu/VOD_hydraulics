# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:09:28 2020

@author: yanlan
"""
import numpy as np
from scipy.stats import mode
import calendar
from scipy import fft
import statsmodels.api as sm

def Gini_simpson(a):
    a = np.reshape(a,[-1,])
    unique_a = np.unique(a)
    pi = np.array([sum(a==itm) for itm in unique_a])/len(a)
    return 1-np.sum(pi**2),mode(a)[0][0]

def LatLon(r,c,dd=0.25,lat0=90,lon0=-180):
    lat  = lat0-r*dd-dd/2
    lon = lon0+c*dd+dd/2
    return lat,lon

def LatLon_r(lat,lon,dd=0.25,lat0=90,lon0=-180):
    r = int(round((90-dd/2-lat)/dd))
    c = int(round((lon-(-180+dd/2))/dd))
    return r,c

def rmfield( a, *fieldnames_to_remove ):
    return a[ [ name for name in a.dtype.names if name not in fieldnames_to_remove ] ]

def interpolate_nan(A):
    nanfilter = ~np.isnan(A)
    np.interp(np.arange(len(A)),np.where(nanfilter)[0],A[nanfilter])
    if np.isnan(sum(A)):
        idx = np.where(nanfilter)[0]
        A[:idx[0]]= A[idx[0]]
        A[idx[-1]:] = A[idx[-1]]
    return A

def getDOY(itm):
    itm_tuple = itm.timetuple()
    return itm_tuple.tm_yday+itm_tuple.tm_hour/24+itm_tuple.tm_min/1440

def CutBounds(A,bounds):
    A[A<bounds[0]] = bounds[0]
    A[A>bounds[1]] = bounds[1]
    return A

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    y = interpolate_nan(y)
    from math import factorial 
    y = interpolate_nan(y)
    window_size = np.abs(np.int(window_size))
    order = np.abs(np.int(order))
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid') 


def toTimestamp(d_list): # for interpolation
  return np.array([calendar.timegm(d.timetuple()) for d in d_list])

def dailyAvg(data,windowsize):
    data = np.array(data)
    data = data[0:windowsize*int(len(data)/windowsize)]
    return np.nanmean(np.reshape(data,[int(len(data)/windowsize),windowsize]),axis=1)


def IsOutlier(a,multiplier=3):
    return np.abs(a-np.nanmedian(a))>multiplier*np.nanstd(a)

    
def calR2(y,yhat):
    mask = ~np.isnan(y+yhat)
    y = y[mask]
    yhat = yhat[mask]
    return 1-np.sum((y-yhat)**2)/np.sum((y-np.mean(y))**2)

def MovAvg(a,windowsize):
    mva = np.zeros(a.shape)+np.nan
    hw = int(windowsize/2)
    for i in range(hw,len(a)-hw):
        mva[i] = np.nanmean(a[i-hw:i+hw])
    return mva

def normalize(A):
    return (A-np.nanmean(A))/np.nanstd(A)

# #import matplotlib.pyplot as plt
# def Fourier_transform(A,T,plotfigure=True,alpha=1,label=''):
#     N = len(A)
#     xf = np.arange(1,(N+1)//2)/N
#     xf = xf/T
#     yf = fft(A)/N
#     Ef = np.abs(yf[1:N//2])**2
#     if plotfigure:
#         plt.plot(xf, Ef,alpha=alpha,label=label)
#         plt.yscale('log')
#         plt.xscale('log')
#         plt.xlabel('f')
#         plt.ylabel('E(f)')
#     return xf,yf,Ef

def itp_rm_outlier(tmpdf,rm = 2):
    tmp = np.array(tmpdf)
    if rm >0: tmp[np.abs(tmp-np.nanmean(tmp))>rm*np.nanstd(tmp)] = np.nan
    x = np.arange(len(tmp))
    xp = x[~np.isnan(tmp)]
    fp = tmp[~np.isnan(tmp)]
    y = np.interp(x,xp,fp)
    return y

def itp(tmpdf):
    tmp = np.array(tmpdf)
    x = np.arange(len(tmp))
    xp = x[~np.isnan(tmp)]
    fp = tmp[~np.isnan(tmp)]
    y = np.interp(x,xp,fp)
    return y

def rm_outlier(tmpdf):
    tmp = np.array(tmpdf)
    tmp[np.abs(tmp-np.nanmean(tmp))>2*np.nanstd(tmp)] = np.nan
    return tmp

def rm_outlier_sn(tmpdf,sn):
    tmp = np.array(tmpdf)
    ncycle = int(len(tmpdf)/len(sn))
    sns = np.tile(sn,[1,ncycle])[0,:]
    anomaly = tmp-sns
    tmp[np.abs(anomaly-np.nanmean(anomaly))>2*np.nanstd(anomaly)] = np.nan
    return tmp

def nanOLS(X,y):
    nanfilter = ~np.isnan(np.sum(X,axis=1)+y)
    if sum(nanfilter)>3:
        X = sm.add_constant(X)
        mod = sm.OLS(y[nanfilter], X[nanfilter])
        res = mod.fit()
    else:
        res = 0
    return res

def cdfmatching(y,yhat): # specifically for variables between 0 and 1 with resolution of .01
    bins = np.arange(0,1,0.01)
    counts, bin_edges = np.histogram(y, bins=bins, normed=True)
    cdf1 = np.cumsum(counts)/sum(counts)
    counts, bin_edges = np.histogram(yhat, bins=bins, normed=True)
    cdf2 = np.cumsum(counts)/sum(counts) 
    yhat_matched = np.array([bin_edges[np.abs(cdf1-cdf2[int(itm*100)]).argmin()] for itm in yhat])
    return yhat_matched