File created by Yanlan Liu. Last update December 2020
Please acknowledge the use of this data in any publications: Liu, Y., N.M. Holtzman, A.G. Konings (2020). Global ecosystem-scale plant hydraulic traits retrieved using model-data fusion. Hydrology and Earth System Sciences Discussions.

The five files correspond to five plant hydraulic traits derived using a model-data fusion approach that integrates a plant hydraulic model and remote sensing derived datasets of vegetation optical depth, surface soil moisture, and evapotranspiration

Resolution: 0.25-degree
Grid: latitude-longitude

Files/variables:
(1) MDF_P50.nc
	lat – center of latitudinal grid cell
	lon – center of longitudinal grid cell
	P50_m - ensemble mean of P50 (the leaf water potential at 50% of xylem conductance, MPa)
	P50_std – standard deviation of P50 across ensembles

(2) MDF_g1.nc
	lat – center of latitudinal grid cell
	lon – center of longitudinal grid cell
	g1_m - ensemble mean of g1 (the slope parameter in Medlyn's stomatal conductance model, kPa^{1/2})
	g1_std – standard deviation of g1 across ensembles

(3) MDF_gpmax.nc
	lat – center of latitudinal grid cell
	lon – center of longitudinal grid cell
	gpmax_m - ensemble mean of gpmax (maximum xylem conductance mm/hr/MPa)
	gpmax_std – standard deviation of gpmax across ensembles

(5) MDF_C.nc
	lat – center of latitudinal grid cell
	lon – center of longitudinal grid cell
	C_m - ensemble mean of C (vegetation capacitance mm/MPa)
	C_std – standard deviation of C across ensembles

(6)	MDF_P50s_P50x.nc
	lat – center of latitudinal grid cell
	lon – center of longitudinal grid cell
	P50s_P50x_m - ensemble mean of P50s/P50x (the ratio between leaf water potential at 50% of stomatal conductance and that at 50% of xylem conductance, unitless)
	P50s_P50x_std – standard deviation of P50s/P50x across ensembles


-------------------------------------------------------------
# Example Python code to read the files:

import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt

f = Dataset('MDF_g1.nc',mode='r') 
lat = np.array(f['lat'][:])
lon = np.array(f['lon'][:])
print(f.variables)
plt.figure(figsize=(14.4,6))
plt.pcolormesh(lon,lat,f.variables['g1_m']);plt.colorbar()