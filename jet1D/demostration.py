import numpy as np
import pandas as pd
import xarray as xr
import glob
import math
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cmocean
from matplotlib.colors import DivergingNorm

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
    """Calculate weights for a low pass Lanczos filter.

    Args:

    window: int
        The length of the filter window.

    cutoff: float
        The cutoff frequency in inverse time steps.

    """
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


def getvalues(dmean):
    b = dmean
    a = dmean.where(dmean==dmean.max(dim='lat'))
    lat=[]
    ua=[]
    width=[]
    #print('is this slow')
    for j in range(len(a.time)):
        for k in range(len(a[j].values)):
            if math.isnan(a[j].values[k])==False:
                lat.append(a[j].lat.values[k])
                ua.append(a[j].values[k])
                fwhm=[]
                for m in range(len(b[j].values)):
                    if b[j].values[m] >= (a[j].values[k])/2:
                        fwhm.append(b[j].lat.values[m])
        if len(fwhm) >= 2:
            width.append([min(fwhm),max(fwhm)])
        else:
            width.append(np.nan)
    return([lat,ua,width])



filenames = glob.glob("/media/peter/Storage/data/native/bigset/uwind/*nc")
filenames.sort()
models=[]
for name in filenames:
    name = name.split('_', 3)[1]  #may need adjusting if directory changes
    models.append(name)


models.sort()

models = models[16]
filenames = filenames[16] #use HadGEM2-CC



x = filenames
index = models

d = xr.open_dataset(x)
d = d.where(d.time==d.time[18030:18390]) # HadGEM2-CC uses silly cf time
#d= d.sel(time=slice('1996-01-01', '1996-12-31'))
dhigh = d.loc[dict(plev=d.plev.values[4])]
dhigh = dhigh.ua
dhigh = dhigh[20:-21]


d = d.loc[dict(plev=d.plev.values[1:3])]
d = get_pressure_weighted(d.ua)
dnofilter = d  #!
dnofilter = dnofilter[20:-21]


wgts = low_pass_weights(41, 1/10)
weight = xr.DataArray(list(wgts), dims=['window'])
d10day = d.rolling(time=41, center=True).construct('window').dot(weight)
d10day = d10day[20:-21]


dmean = d10day.mean(dim='lon')
values = getvalues(dmean)



vmin=-30
vmax=80
cmap = cmocean.cm.delta


for i in range(len(dhigh.time)):
    fig = plt.figure(figsize=(16, 9))
    plt.gca().set_visible(False)
    gs = gridspec.GridSpec(nrows=9,
                           ncols=9,
                           figure=fig,
                           width_ratios= [1, 1, 1, 1, 1, 1, 1, 1, 1],
                           height_ratios=[1, 1, 1, 1, 1, 1, 1, 1, 1],
                           wspace=0.1,
                           hspace=0.1)
    ax1 = fig.add_subplot(gs[1:8,0:3], projection=ccrs.PlateCarree())
    ax1.coastlines()
    z = dhigh.sel(time=dhigh.time[i].values).plot(ax=ax1,vmin=vmin, vmax=vmax,cmap=cmap, norm=DivergingNorm(0),add_colorbar=False)
    plt.gca().outline_patch.set_visible(False)
    plt.title('250 hPa', fontsize=16, ha='center')
    ax2 = fig.add_subplot(gs[1:8,3:6], projection=ccrs.PlateCarree())
    ax2.coastlines()
    dnofilter.sel(time=dnofilter.time[i].values).plot(ax=ax2,vmin=vmin, vmax=vmax,cmap=cmap, norm=DivergingNorm(0),add_colorbar=False)
    plt.gca().outline_patch.set_visible(False)
    plt.title('850-700 hPa Pressure Weighted', fontsize=16, ha='center')
    plt.text(0.5, 1.08,str(models)+' Zonal Wind \n '+str(dhigh.time[i].values), fontsize=16, ha='center')
    ax3 = fig.add_subplot(gs[1:8,6:9], projection=ccrs.PlateCarree())
    ax3.coastlines()
    d10day.sel(time=d10day.time[i].values).plot(ax=ax3,vmin=vmin, vmax=vmax,cmap=cmap, norm=DivergingNorm(0),add_colorbar=False)
    plt.axhline(y=values[0][i],color='r', linestyle='-')
    plt.text(35, values[0][i], str(round(values[1][i],1)))
    plt.axhline(y=values[2][i][0],color='r', linestyle='dashed')
    plt.axhline(y=values[2][i][1],color='r', linestyle='dashed')
    plt.gca().outline_patch.set_visible(False)
    plt.title('Lanczos Filter Applied', fontsize=16, ha='center')
    ax4 = fig.add_subplot(gs[8:9,0:9])
    img = dhigh.sel(time=dhigh.time[i].values).plot(ax=ax4,vmin=vmin, vmax=vmax,cmap=cmap, norm=DivergingNorm(0),add_colorbar=False)
    plt.gca().set_visible(False)
    cax = plt.axes([0.12, 0.25, 0.789, 0.03])
    plt.colorbar(z,orientation= 'horizontal', cax=cax)
    plt.text(27, -140,'m/s', fontsize=16, ha='center')
    plt.savefig('../../JET_STREAM_OUT/jet1D/demonstration/'+str(i)+'_togif.png')
    plt.close()

#cat `ls -v *togif.png` | ffmpeg -framerate 5 -f image2pipe -i - output.mp4
