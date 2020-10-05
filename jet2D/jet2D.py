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
    x = x.resample(time='QS-DEC').mean(dim='time',skipna=True)
    return x


def rainref(pr,x):
    pr = pr.sel(lat=-34,method='nearest')
    pr = pr.sel(lon=18,method='nearest')
    pr = pr.resample(time='QS-DEC').mean(dim='time',skipna=True)
    pr = pr.sel(time=list(x.time.values))
    return pr

def correlatornator(x,pr):
    x = x.dropna(dim='time')
    pr = pr.dropna(dim='time')
    cor = x.copy()
    cor = cor[0]*0
    cor = cor.drop_vars('time')
    pv = cor.copy()
    for i in range(len(x.lat)):
        for j in range(len(x.lon)):
            cor[i][j] = scipy.stats.pearsonr(list(x.sel(lat=x.lat[i]).sel(lon=x.lon[j]).values),list(pr.values))[0]
            pv[i][j] = scipy.stats.pearsonr(list(x.sel(lat=x.lat[i]).sel(lon=x.lon[j]).values),list(pr.values))[1]
    return cor,pv


filenames = glob.glob("/media/peter/C6ECF47AECF46659/data/2deg/*") #file locations
filenames.sort()

models=[]
for name in filenames:
    models.append(name.split('/')[-1].split('_')[1].split('.')[0])

models.sort()



### do the jet stream outline for CMIP5
jetdic={}
prdic={}
cordic={}
pvdic={}
for file, model in zip(filenames, models):
    x =xr.open_dataset(file)
    x = x.rename({'__xarray_dataarray_variable__':'ws'})
    x = x.sel(plev=slice(85000.0,70000.0))
    x = get_pressure_weighted(x)
    x = x.sel(time=slice('1950', '2005'))
    jetdic[model] = jetenator(x)
    pr =xr.open_dataset('/media/peter/C6ECF47AECF46659/data/native/subset/pr/subset_'+str(model)+'_pr.nc')
    pr = pr = pr.sel(time=slice('1950', '2005'))
    prdic[model] = rainref(pr,jetdic[model])
    cordic[model]={}
    pvdic[model] = {}
    for seas in ['DJF','MAM','JJA','SON']:
        cordic[model][seas],pvdic[model][seas] = correlatornator(jetdic[model].where(jetdic[model].time.dt.season==seas).dropna(dim='time'),prdic[model].pr.where(prdic[model].time.dt.season==seas).dropna(dim='time'))


### do the jet stream outline for NOAA
x =xr.open_dataset('/media/peter/C6ECF47AECF46659/data/NOAA_ws_levels.nc')
x = x.rename({'__xarray_dataarray_variable__':'ws'})
x = x.rename({'level':'plev'})
x = x.sel(time=slice('1950', '2005'))
x = x.where(x.lat==jetdic[model].lat)
x = x.where(x.lon==jetdic[model].lon)
x = x.sel(plev=slice(850,700))
x = get_pressure_weighted(x)
jetdic['NOAA'] = jetenator(x)
pr =xr.open_dataset('/media/peter/C6ECF47AECF46659/data/NOAA_pr.nc')
pr = pr = pr.sel(time=slice('1950', '2005'))
prdic['NOAA'] = rainref(pr,jetdic['NOAA'])
model='NOAA'
cordic[model]={}
pvdic[model] = {}
for seas in ['DJF','MAM','JJA','SON']:
    cordic[model][seas],pvdic[model][seas] = correlatornator(jetdic[model].where(jetdic[model].time.dt.season==seas).dropna(dim='time'),prdic[model].prate.where(prdic[model].time.dt.season==seas).dropna(dim='time'))

### do the jet stream outline for ERA5
x =xr.open_dataset('/media/peter/C6ECF47AECF46659/data/ERA5_ws.nc')
x = x.rename({'__xarray_dataarray_variable__':'ws'})
x = x.rename({'level':'plev'})
x = x.rename({'longitude':'lon'})
x = x.rename({'latitude':'lat'})
x = x.sel(time=slice('1980', '2018'))
x = x.sel(plev=slice(700.0,850.0))
x = x.where(x.lat==jetdic[model].lat)
x = x.where(x.lon==jetdic[model].lon)
x = get_pressure_weighted(x)
jetdic['ERA5'] = jetenator(x)
pr =xr.open_dataset('/media/peter/C6ECF47AECF46659/data/ERA5_pr_1hr_1979-2019.nc')
pr = pr.sel(time=slice('1980', '2018'))
prdic['ERA5'] = rainref(pr,jetdic['ERA5'])
cordic['ERA5'],pvdic['ERA5'] = correlatornator(jetdic['ERA5'],prdic['ERA5'].pr)


jetdic['ERA5_2005'] = jetenator(x.sel(time=slice('1980', '2005')))
pr =xr.open_dataset('/media/peter/C6ECF47AECF46659/data/ERA5_pr_1hr_1979-2019.nc')
pr = pr.sel(time=slice('1980', '2005'))
prdic['ERA5_2005'] = rainref(pr,jetdic['ERA5_2005'])
cordic['ERA5_2005'],pvdic['ERA5_2005'] = correlatornator(jetdic['ERA5_2005'],prdic['ERA5_2005'].pr)



cormax = max((np.nanmax(cordic[key].values) for key in cordic))
cormin = min((np.nanmin(cordic[key].values) for key in cordic))
levels = np.linspace(round(-0.7,1),round(0.7,1), 15)

from matplotlib import cm

for index in jetdic:
index = 'NOAA'

for season in ['DJF','MAM','JJA','SON']:
    ax = plt.axes(projection=ccrs.Orthographic())
    ax.coastlines()
    plt.rcParams['hatch.linewidth']=0.4
    plt.rcParams['hatch.color']='black'
    c = ax.contourf(cordic[index][season].lon,cordic[index][season].lat,cordic[index][season],levels=levels, transform = ccrs.PlateCarree(),alpha=0.6)
    sig = ax.contourf(pvdic[index][season].lon,pvdic[index][season].lat,pvdic[index][season],levels = [0,0.05], hatches=["/"], transform = ccrs.PlateCarree(),alpha=0)
    prob = ax.contour(jetdic[index].lon,jetdic[index].lat,jetdic[index].where(jetdic[index].time.dt.season==season).mean(dim='time'),vmin=0,vmax=1,colors = 'snow',levels = 5,linewidths=np.linspace(0.4,3.5,5), transform = ccrs.PlateCarree(),alpha=1)
    ax.clabel(prob, inline=True, fontsize=9)
    plt.title(str(season))
    ax.set_extent([-50, 32, -15, -85], ccrs.PlateCarree())
    plt.colorbar(c)
    ax.gridlines(linewidth=0.5, color='gray', alpha=0.5)
    plt.savefig(str(index)+'_Jet_correlation_Seasonal_'+str(season)+'.pdf')
    plt.close()
    #plt.savefig('correlationplots_2deg/'+str(index)+'_Jet_correlation_Seasonal.pdf')


#mean absolute error calc
results=[]
for index in jetdic:
    MAE=[]
    for season in ['DJF','MAM','JJA','SON']:
        ref = jetdic['NOAA'].where(jetdic['NOAA'].time.dt.season==season).mean(dim='time')
        x = jetdic[index].where(jetdic[index].time.dt.season==season).mean(dim='time')
        for i in ref.lat.values:
            for j in ref.lon.values:
                MAE.append(float(np.abs(ref.sel(lat=i).sel(lon=j) - x.sel(lat=i).sel(lon=j))))
    results.append([index,np.mean(MAE)])


results = pd.DataFrame(results,columns=['Model','score'])
results = results.sort_values('score')
results.to_csv('scores_seasonal_decomposed.csv')
