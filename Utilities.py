# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:09:28 2020

@author: yanlan
"""
import numpy as np
import calendar


def LatLon(r,c,dd=0.25,lat0=90,lon0=-180):
    lat  = lat0-r*dd-dd/2
    lon = lon0+c*dd+dd/2
    return lat,lon

def LatLon_r(lat,lon,dd=0.25,lat0=90,lon0=-180):
    r = int(round((90-dd/2-lat)/dd))
    c = int(round((lon-(-180+dd/2))/dd))
    return r,c


def interpolate_nan(A):
    nanfilter = ~np.isnan(A)
    A = np.interp(np.arange(len(A)),np.where(nanfilter)[0],A[nanfilter])
    if np.isnan(sum(A)):
        idx = np.where(nanfilter)[0]
        A[:idx[0]]= A[idx[0]]
        A[idx[-1]:] = A[idx[-1]]
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

def dailyMin(data,windowsize):
    data = np.array(data)
    data = data[0:windowsize*int(len(data)/windowsize)]
    return np.nanmin(np.reshape(data,[int(len(data)/windowsize),windowsize]),axis=1)


def IsOutlier(a,multiplier=3):
    return np.abs(a-np.nanmedian(a))>multiplier*np.nanstd(a)


def MovAvg(a,windowsize):
    mva = np.zeros(a.shape)+np.nan
    hw = int(windowsize/2)
    for i in range(hw,len(a)-hw):
        mva[i] = np.nanmean(a[i-hw:i+hw])
    return mva


def rm_outlier(tmpdf):
    tmp = np.array(tmpdf)
    tmp[np.abs(tmp-np.nanmean(tmp))>3*np.nanstd(tmp)] = np.nan
    return tmp


def cdfmatching(y,yhat): # specifically for variables between 0 and 1 with resolution of .01
    bins = np.arange(0,1,0.01)
    counts, bin_edges = np.histogram(y, bins=bins, normed=True)
    cdf1 = np.cumsum(counts)/sum(counts)
    counts, bin_edges = np.histogram(yhat, bins=bins, normed=True)
    cdf2 = np.cumsum(counts)/sum(counts) 
    yhat_matched = np.array([bin_edges[np.abs(cdf1-cdf2[int(itm*100)]).argmin()] for itm in yhat])
    return yhat_matched
