#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 21:39:00 2020

@author: yanlan
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="ticks", color_codes=True,font_scale=1.75)
import pickle
#%%
fname = './Data/ISLSCP_C4_1DEG_932/data/c4_percent_1d.asc'
A = np.loadtxt(fname,skiprows=6)

#%%
plt.figure(figsize=(13.5,5))
plt.imshow(A[40:65,50:118],cmap='Greens')
plt.title('C4 fraction (%)')
plt.colorbar();plt.xticks([]);plt.yticks([]);plt.clim([0,100])
