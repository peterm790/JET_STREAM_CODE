import numpy as np
import pandas as pd
import glob
import math
import statistics
import scipy.stats as stats
import pickle
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import xarray as xr
from sklearn.cluster import KMeans


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

dic = pickle.load( open( "daily_jettrack.p", "rb" ) )

dic['ERA5_2005'] = dic['ERA5'].where(dic['ERA5'].years<=2005).dropna()

def is_winter(month):
    return (month == 6) | (month == 7) | (month == 8)# | (month == 5) | (month == 9) | (month == 4)


monthly={}
annual={}
winter={}
seasonal={}
for index in dic:
    annual[index] = pd.DataFrame(pd.to_numeric(dic[index].set_index('years').lat).groupby(level='years').mean())
    annual[index]['ua'] = pd.to_numeric(dic[index].set_index('years').ua).groupby(level='years').mean()
    annual[index]['width'] = pd.to_numeric(dic[index].set_index('years').width).groupby(level='years').mean()
    annual[index]['pr'] = pd.to_numeric(dic[index].set_index('years').pr).groupby(level='years').mean()
    monthly[index] = pd.DataFrame(pd.to_numeric(dic[index].set_index(['years','months']).lat).groupby(['years','months']).mean())
    monthly[index]['ua'] = pd.to_numeric(dic[index].set_index(['years','months']).ua).groupby(['years','months']).mean()
    monthly[index]['width'] = pd.to_numeric(dic[index].set_index(['years','months']).width).groupby(['years','months']).mean()
    monthly[index]['pr'] = pd.to_numeric(dic[index].set_index(['years','months']).pr).groupby(['years','months']).mean()
    winter[index] = pd.DataFrame(pd.to_numeric(dic[index].where(is_winter(dic[index].months)).dropna().set_index('years').lat).groupby(level='years').mean())
    winter[index]['ua'] = pd.to_numeric(dic[index].where(is_winter(dic[index].months)).dropna().set_index('years').ua).groupby(level='years').mean()
    winter[index]['width'] = pd.to_numeric(dic[index].where(is_winter(dic[index].months)).dropna().set_index('years').width).groupby(level='years').mean()
    winter[index]['pr'] = pd.to_numeric(dic[index].where(is_winter(dic[index].months)).dropna().set_index('years').pr).groupby(level='years').mean()
    seasonal[index] = pd.DataFrame(pd.to_numeric(dic[index].set_index(['years','seasons']).lat).groupby(level=['years','seasons']).mean())
    seasonal[index]['ua'] = pd.to_numeric(dic[index].set_index(['years','seasons']).ua).groupby(level=['years','seasons']).mean()
    seasonal[index]['width'] = pd.to_numeric(dic[index].set_index(['years','seasons']).width).groupby(level=['years','seasons']).mean()
    seasonal[index]['pr'] = pd.to_numeric(dic[index].set_index(['years','seasons']).pr).groupby(level=['years','seasons']).mean()

df =[]
df = seasonal['NOAA']

df2=df.copy()

y = df.pr.to_numpy()
# Create linear regression object
regr = linear_model.LinearRegression()
X = df[['lat','ua','width']].to_numpy()
regr.fit(X, y)
df2['all3'] = regr.predict(X)
# Create linear regression object
regr = linear_model.LinearRegression()
X = df[['lat','width']].to_numpy()
regr.fit(X, y)
df2['width_lat'] = regr.predict(X)
# Create linear regression object
regr = linear_model.LinearRegression()
X = df[['ua','width']].to_numpy()
regr.fit(X, y)
df2['width_ua'] = regr.predict(X)
# Create linear regression object
regr = linear_model.LinearRegression()
X = df[['lat']].to_numpy()
regr.fit(X, y)
df2['lat_pred'] = regr.predict(X)
# Create linear regression object
regr = linear_model.LinearRegression()
X = df[['width']].to_numpy()
regr.fit(X, y)
df2['width_pred'] = regr.predict(X)
# Create linear regression object
regr = linear_model.LinearRegression()
X = df[['ua']].to_numpy()
regr.fit(X, y)
df2['ua_pred'] = regr.predict(X)





x =xr.open_dataset('/media/peter/C6ECF47AECF46659/data/NOAA_ws_levels.nc')
x = x.rename({'__xarray_dataarray_variable__':'ws'})
x = x.rename({'level':'plev'})
x = x.sel(time=slice('1951', '2004'))
x = x.sel(plev=slice(850,700))
x = get_pressure_weighted(x)
x = jetenator(x)
x = x[:-1]
X = np.array(x)
X = X.reshape(X.shape[0], X.shape[-2]*X.shape[-1])

kmeans = KMeans(n_clusters=3, random_state=2).fit_predict(X.T)
clusters = kmeans.reshape(len(x.lat.values),len(x.lon.values))
xm = np.array(x.mean(dim='time'))
xxr = xr.Dataset({ "jet" : (("time", "lat", "lon"),x,),"clust": (("lat", "lon"), clusters),},coords={"lat": x.lat.values, "lon": x.lon.values,"time":x.time.values})
Xc = np.array(xxr.jet.groupby(xxr.clust).mean())
regr = linear_model.LinearRegression()
regr.fit(Xc, y)
df2['twoD3C'] = regr.predict(Xc)


x =xr.open_dataset('/media/peter/C6ECF47AECF46659/data/NOAA_ws_levels.nc')
x = x.rename({'__xarray_dataarray_variable__':'ws'})
x = x.rename({'level':'plev'})
x = x.sel(time=slice('1951', '2004'))
x = x.sel(plev=slice(850,700))
x = get_pressure_weighted(x)
x = x.fillna(0)
x = x.resample(time='QS-DEC').mean(dim='time',skipna=True).ws
x = x[:-1]
X = np.array(x)
X = X.reshape(X.shape[0], X.shape[-2]*X.shape[-1])

kmeans = KMeans(n_clusters=3, random_state=2).fit_predict(X.T)
clusters = kmeans.reshape(len(x.lat.values),len(x.lon.values))
xm = np.array(x.mean(dim='time'))
xxr = xr.Dataset({ "jet" : (("time", "lat", "lon"),x,),"clust": (("lat", "lon"), clusters),},coords={"lat": x.lat.values, "lon": x.lon.values,"time":x.time.values})
Xc = np.array(xxr.jet.groupby(xxr.clust).mean())
regr = linear_model.LinearRegression()
regr.fit(Xc[:-20], y[:-20])
df2['mean3C']
mean3c= regr.predict(Xc[-20:])

r2_score(df2.pr[-20:], mean3c)




fig,ax = plt.subplots(figsize=(16,9))
df2.pr[-40:].plot(color='blue',label='pr',lw=3)
df2.all3[-40:].plot(color='green',label='Width + Latitude + Velocity ['+str(np.round(r2_score(df2.pr, df2.all3),4))+']',lw=2.7,alpha=0.9)
df2.width_lat[-40:].plot(color='red',label='Width + Latitude ['+str(np.round(r2_score(df2.pr, df2.width_lat),4))+']',lw=2.7,alpha=0.9)
df2.width_ua[-40:].plot(color='deeppink',label='Width + Velocity ['+str(np.round(r2_score(df2.pr, df2.width_ua),4))+']',lw=2.7,alpha=0.9)
df2.width_pred[-40:].plot(color='black',label='Width ['+str(np.round(r2_score(df2.pr, df2.width_pred),4))+']',lw=2.7,alpha=0.9)
df2.lat_pred[-40:].plot(color='orange',label='Latitude ['+str(np.round(r2_score(df2.pr, df2.lat_pred),4))+']',lw=2.7,alpha=0.9)
df2.ua_pred[-40:].plot(color='yellow',label='Velocity ['+str(np.round(r2_score(df2.pr, df2.ua_pred),4))+']',lw=2.7,alpha=0.9)
df2.twoD3C[-40:].plot(color='purple',label='2D 3 clusters ['+str(np.round(r2_score(df2.pr, df2.twoD3C),4))+']',lw=2.7,alpha=0.9)
df2.mean3C[-40:].plot(color='grey',label='Mean WS 3 clusters ['+str(np.round(r2_score(df2.pr, df2.mean3C),4))+']',lw=2.7,alpha=0.9)
plt.legend(loc=1)
plt.title('Linear Regression')
ax.set_ylabel("Seasonal Mean Precipitation [kg/m^2/s]")
plt.savefig('linear_reg_2D.png',dpi=100)
plt.show()





fig, ax1 = plt.subplots(figsize=(10,6))
sns.kdeplot(list(df.pr),lw=2.5,color='b', vertical=True, label='Model pr',ax=ax1)
sns.kdeplot(list(df.all3),lw=2.5,color='g', vertical=True, label='all 3',ax=ax1)
sns.kdeplot(list(df.all2),lw=2.5,color='r', vertical=True, label='all 2',ax=ax1)
sns.kdeplot(list(df.lat_pred),lw=2.5,color='grey', vertical=True, label='lat',ax=ax1)
sns.kdeplot(list(df.width_pred),lw=2.5,color='black', vertical=True, label='width',ax=ax1)
sns.kdeplot(list(df.ua_pred),lw=2.5,color='y', vertical=True, label='ua',ax=ax1)
plt.show()



for col in df.columns[6:]:
    y_pred = df[col]
    y = df['pr']
    print(col)
    # The mean squared error
    print('Mean squared error:' ,mean_squared_error(y, y_pred))
    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination: %.4f' % r2_score(y, y_pred))
