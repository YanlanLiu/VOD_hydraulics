#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 22:06:00 2020

@author: yanlan
"""

from matplotlib import pyplot as plt
import pandas as pd
import netCDF4
fp='../Trugman_map/CWM_P50_10Deg.nc'
nc = netCDF4.Dataset(fp)
plt.imshow(np.flipud(np.transpose(nc['CWM_P50'][:,:])));plt.colorbar()
plt.show()