### import the things
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import glob
import os
import scipy.stats


def get_pressure_weighted(x):
    dPref = (x.plev.values[0]-x.plev.values[-1])  #(p-ps)
    integral = []
    for i in range(len(x.plev)): #Integral of variable from P to Ps calculated as area between each pressure variable trapezoid then summed
        if i+1 < len(x.plev):
            area=((x.loc[dict(plev=x.plev.values[i])] + x.loc[dict(plev=x.plev.values[i+1])])/2)*(x.plev.values[i]-x.plev.values[i+1])
            integral.append(area)
    pw = (sum(integral))/dPref
    return(pw)


filenames = glob.glob("/media/peter/C6ECF47AECF46659/data/2deg/*") #file locations
filenames.sort()

models=[]
for name in filenames:
    models.append(name.split('/')[-1].split('_')[1].split('.')[0])

models.sort()


dic={}
for file, model in zip(filenames, models):
    x =xr.open_dataset(file)
    x = x.rename({'__xarray_dataarray_variable__':'ws'})
    x = x.sel(plev=slice(85000.0,70000.0))
    x = get_pressure_weighted(x)
    x = x.sel(time=slice('1950', '2005'))
    x = x.fillna(0)
    dic[model]={}
    for seas in ['DJF','MAM','JJA','SON']:
        dic[model][seas] = x.where(x.time.dt.season==seas).dropna(dim='time').mean(dim='time',skipna=True).ws


x =xr.open_dataset('/media/peter/C6ECF47AECF46659/data/NOAA_ws_levels.nc')
x = x.rename({'__xarray_dataarray_variable__':'ws'})
x = x.rename({'level':'plev'})
x = x.sel(time=slice('1950', '2005'))
x = x.where(x.lat==dic[model][seas].lat)
x = x.where(x.lon==dic[model][seas].lon)
x = x.sel(plev=slice(850,700))
x = get_pressure_weighted(x)
x = x.fillna(0)
dic['NOAA']={}
for seas in ['DJF','MAM','JJA','SON']:
    dic['NOAA'][seas] = x.where(x.time.dt.season==seas).dropna(dim='time').mean(dim='time',skipna=True).ws


x =xr.open_dataset('/media/peter/C6ECF47AECF46659/data/ERA5_ws.nc')
x = x.rename({'__xarray_dataarray_variable__':'ws'})
x = x.rename({'level':'plev'})
x = x.rename({'longitude':'lon'})
x = x.rename({'latitude':'lat'})
x = x.sel(time=slice('1980', '2018'))
x = x.sel(plev=slice(700.0,850.0))
x = x.where(x.lat==dic[model][seas].lat)
x = x.where(x.lon==dic[model][seas].lon)
x = get_pressure_weighted(x)
x = x.fillna(0)
dic['ERA5']={}
dic['ERA5_2005']={}
for seas in ['DJF','MAM','JJA','SON']:
    dic['ERA5'][seas] = x.where(x.time.dt.season==seas).dropna(dim='time').mean(dim='time',skipna=True).ws
    dic['ERA5_2005'][seas] = x.where(x.time.dt.season==seas).dropna(dim='time').sel(time=slice('1980', '2005')).mean(dim='time',skipna=True).ws

#mean absolute error calc
results=[]
for index in dic:
    MAE=[]
    for season in ['DJF','MAM','JJA','SON']:
        ref = dic['NOAA'][seas]
        x = dic[index][seas]
        for i in ref.lat.values:
            for j in ref.lon.values:
                MAE.append(float(np.abs(ref.sel(lat=i).sel(lon=j) - x.sel(lat=i).sel(lon=j))))
    results.append([index,np.mean(MAE)])



results = pd.DataFrame(results,columns=['Model','score'])
results = results.sort_values('score')

results.to_csv('scores_mean_decomp.csv')
