### import the things
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import glob
import os
import scipy.stats
from matplotlib import cm

data_path  = '/media/peter/Storage/data/'


def get_pressure_weighted(x):
    dPref = (x.plev.values[0]-x.plev.values[-1])  #(p-ps)
    integral = []
    for i in range(len(x.plev)): #Integral of variable from P to Ps calculated as area between each pressure variable trapezoid then summed
        if i+1 < len(x.plev):
            area=((x.loc[dict(plev=x.plev.values[i])] + x.loc[dict(plev=x.plev.values[i+1])])/2)*(x.plev.values[i]-x.plev.values[i+1])
            integral.append(area)
    pw = (sum(integral))/dPref
    return(pw)


def low_pass_weights(window, cutoff):
    order = ((window - 1) // 2 ) + 1
    nwts = 2 * order + 1
    w = np.zeros([nwts])
    n = nwts // 2
    w[n] = 2 * cutoff
    k = np.arange(1., n)
    sigma = np.sin(np.pi * k / n) * n / (np.pi * k)
    firstfactor = np.sin(2. * np.pi * cutoff * k) / (np.pi * k)
    w[n-1:0:-1] = firstfactor * sigma
    w[n+1:-1] = firstfactor * sigma
    return w[1:-1]

wgts = low_pass_weights(41, 1/10)
weight = xr.DataArray(list(wgts), dims=['window'])

def jetenator(x):
    x = x.fillna(0)
    x = x.ws.rolling(time=41, center=True).construct('window').dot(weight)
    x = x.dropna(dim='time',how='all')
    limit = np.quantile(x.values,0.9)
    x = x.where((x>=limit))
    x = x/x
    x = x.fillna(0)
    #x = x.resample(time='QS-DEC').mean(dim='time',skipna=True)
    return x



filenames = glob.glob(str(data_path)+"2deg/*") #file locations
filenames.sort()

models=[]
for name in filenames:
    models.append(name.split('/')[-1].split('_')[1].split('.')[0])

models.sort()

model = models[15]
file = filenames[15]

x =xr.open_dataset(file)
x = x.rename({'__xarray_dataarray_variable__':'ws'})
x = x.sel(plev=slice(85000.0,70000.0))
x = x.sel(time=slice('2000', '2000'))
x = get_pressure_weighted(x)
jet = jetenator(x)

for i in range(len(jet.time)):
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.contour(jet.lon,jet.lat,jet[i],colors = 'black', transform = ccrs.PlateCarree(),alpha=1)
    plt.title('HadGEM2-CC \n'+str(jet[i].time.values))
    ax.set_extent([-50, 32, -15, -85], ccrs.PlateCarree())
    ax.gridlines(linewidth=0.5, color='gray', alpha=0.5)
    plt.savefig('../../JET_STREAM_OUT/jet2D/demonstration/'+str(i)+'_togif.png')
    plt.close()


#cat `ls -v *togif.png` | ffmpeg -framerate 6 -f image2pipe -i - output.mp4
