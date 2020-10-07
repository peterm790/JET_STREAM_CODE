import numpy as np
import pandas as pd
import xarray as xr
import glob
import math
import statistics
import pickle


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

wgts = low_pass_weights(41, 1/10)
weight = xr.DataArray(list(wgts), dims=['window'])

def getrange(numbers):
    return max(numbers) - min(numbers)

def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx


def jettracker(d,levels=[85000,70000],time=['1950', '2005'],infer = np.linspace(-75, 15, 501)):
    _, index = np.unique(d['time'], return_index=True)
    d = d.isel(time=index)
    d = d.sel(plev=levels,method='nearest')
    print(d.plev.values)
    d = d.sel(time=slice(time[0], time[-1]))
    #d =d.ua
    d = get_pressure_weighted(d.ua)
    d = d.rolling(time=41, center=True).construct('window').dot(weight)
    d = d[20:-21]
    d = d.mean(dim='lon',skipna=True)
    d = d.sortby(d.lat)
    list = []
    size = len(d.lat)-1
    for y in pd.date_range(str(int(time[0])+1),time[-1], freq='A'): #1980-2004
        a = d.sel(time = str(y.year))
        b = a.where(a==a.max(dim='lat')) #find max zonal mean/set rest to Nan
        for i in range(len(a.values)): #each day
            fwhm=[]
            for j in range(len(a.values[i])): #each lat
                if math.isnan(b.values[i][j])==False: #if a maximum is found
                    if j==0 or j==size:
                        lat = a.lat.values[j]
                        ua = b.values[i][j]
                        break
                    else:
                        p = np.poly1d(np.polyfit(b.lat.values[j-1:j+2],a.values[i][j-1:j+2],2))
                        lat = pd.DataFrame(p(infer),index=infer).idxmax().values[0]
                        ua = pd.DataFrame(p(infer),index=infer).max().values[0]
                        break
            for k in range(len(b.lat.values)): #check against each lat
                if a.values[i][k] >= ua/2:
                    fwhm.append(k)
            if len(fwhm) >= 2: #some days no latitudes other than maximum meet fwhm criteria
                if fwhm[0]==0:
                    bottom = b.lat.values[fwhm[0]]
                else:
                    wb = np.poly1d(np.polyfit(b.lat.values[fwhm[0]-1:fwhm[0]+2],a.values[i][fwhm[0]-1:fwhm[0]+2],1))
                    inf = np.linspace(b.lat.values[fwhm[0]+1],b.lat.values[fwhm[0]-1],100)
                    bottom = inf[find_nearest(wb(inf),ua/2)]
                if fwhm[-1]==size or fwhm[-1]==0:
                    top = b.lat.values[fwhm[-1]]
                else:
                    wt = np.poly1d(np.polyfit(b.lat.values[fwhm[-1]-1:fwhm[-1]+2],a.values[i][fwhm[-1]-1:fwhm[-1]+2],1))
                    inf = np.linspace(b.lat.values[fwhm[-1]+1],b.lat.values[fwhm[-1]-1],100)
                    top = inf[find_nearest(wt(inf),ua/2)]
                width = top - bottom
            else:
                width = np.nan
            list.append([a[i].time.values,lat,ua,width]) #append to list
    df = pd.DataFrame(np.array(list),columns=['time', 'lat', 'ua','width'])
    return(df)


filenames = glob.glob("/media/peter/C6ECF47AECF46659/data/native/bigset/uwind/*.nc") #file locations
filenames.sort()
models=[]
for name in filenames:
    name = name.split('_', 3)[1]  #needs adjusting if directory changes
    models.append(name)

models.sort()


dic={}
for file, model in zip(filenames, models):
    print(model)
    dic[model] = jettracker(xr.open_dataset(file))
    print('done')


x = xr.open_dataset('/media/peter/C6ECF47AECF46659/data/native/bigset/uwind/NOAA/ua_NOAA_final.nc')
x = x.rename({'uwnd':'ua'})
x = x.rename({'level':'plev'})
dic['NOAA'] = jettracker(x,levels=[850,700])


####ERA5 done year by year to reduce ram load
def is_end(month):
    return (month == 12)

def is_start(month):
    return (month == 1)

ERAdf = pd.DataFrame([],columns=['time', 'lat', 'ua','width'])
for i in pd.date_range('1980','2018', freq='A'):
    files = {}
    d = xr.open_dataset('/media/peter/C6ECF47AECF46659/data/native/bigset/uwind/ERA5/daily/daily_ua_ERA5_1979_2019.nc')
    files['now'] = d.sel(time=str(i.year))
    files['old'] = d.sel(time=str(i.year+1))
    files['new'] = d.sel(time=str(i.year-1))
    files['old'] = files['old'].sel(time=is_end(files['old'].time.dt.month)).compute()
    files['new'] = files['new'].sel(time=is_start(files['new'].time.dt.month)).compute()
    d.close()
    x = xr.merge([files['old'], files['now'], files['new']])
    x = x.rename({'latitude':'lat'})
    x = x.rename({'longitude':'lon'})
    x = x.rename({'level':'plev'})
    ERAdf = ERAdf.append(jettracker(x,levels=[850,700],time=[str(i.year-1), str(i.year+1)])) #too little RAM - killed

dic['ERA5'] = ERAdf.reset_index()
#############pr
filenames = glob.glob("/media/peter/C6ECF47AECF46659/data/native/subset/pr/subset*nc")
filenames.sort()
models=[]
for name in filenames:
    name = name.split('_', 3)[1]  #may need adjusting if directory changes
    models.append(name)


models.sort()


for file,model  in zip(filenames,models):
    x = xr.open_dataset(str(file))
    _, index = np.unique(x['time'], return_index=True)
    x = x.isel(time=index)
    x = x.sel(time=slice('1951', '2004'))
    x = x.sel(lat=-34,method='nearest')
    x = x.sel(lon=18,method='nearest')
    x = x.pr.values
    dic[model]['pr'] = x


x = xr.open_dataset('/media/peter/C6ECF47AECF46659/data/NOAA_pr.nc')
_, index = np.unique(x['time'], return_index=True)
x = x.isel(time=index)
x = x.sel(time=slice('1951', '2004'))
x = x.sel(lat=-34,method='nearest')
x = x.sel(lon=18,method='nearest')
x = x.prate.values
dic['NOAA']['pr'] = x


pr = xr.open_dataset('/media/peter/C6ECF47AECF46659/data/ERA5_pr_1hr_1979-2019.nc')
pr = pr.sel(lat=-34,method='nearest')
pr = pr.sel(lon=18,method='nearest')
pr = pr.sel(time=slice('1980', '2017'))
pr = pr.resample(time = 'D').mean(dim='time')
dic['ERA5']['pr'] = pr.pr.values

arraytype = type(dic['bcc-csm1-1-m'].time[0])

for index in dic:
    years=[]
    months=[]
    print(index)
    if isinstance(dic[index].time[0], arraytype):
        print('ugly')
        for i in range(len(dic[index])):
            dic[index].time[i] = dic[index].time[i].ravel()[0]
            years.append(dic[index].time[i].year)
            months.append(dic[index].time[i].month)
        dic[index]['years'] = years
        dic[index]['months'] = months
    else:
        print('noice')
        for i in range(len(dic[index])):
            years.append(dic[index].time[i].year)
            months.append(dic[index].time[i].month)
        dic[index]['years'] = years
        dic[index]['months'] = months



def seasoning(month):
    if (month == 12) | (month == 1) | (month == 2):
        return(1)
    if (month == 3) | (month == 4) | (month == 5):
        return(2)
    if (month == 6) | (month == 7) | (month == 8):
        return(3)
    if (month == 9) | (month == 10) | (month == 11):
        return(4)

for index in dic:
    dic[index]['seasons'] = [seasoning(i) for i in dic[index].months.values]




pickle.dump( dic, open( "daily_jettrack_new.p", "wb" ) )

#dic = pickle.load( open( "daily_jettrack.p", "rb" ) )

#updated Aug 14
