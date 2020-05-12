# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:09:28 2020

@author: yanlan
"""
import numpy as np
from scipy.stats import mode
import calendar


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

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    from math import factorial 
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

def calR2(y,yhat):
    mask = ~np.isnan(y+yhat)
    y = y[mask]
    yhat = yhat[mask]
    return 1-np.sum((y-yhat)**2)/np.sum((y-np.mean(y))**2)