#le grand table
import numpy as np
import pandas as pd
import xarray as xr
import glob
import math
import statistics
import pickle
import scipy.stats

# rework to rely on base of the daily data from a common function

dic = pickle.load( open( "../../JET_STREAM_OUT/jet1D/jettrack.p", "rb" ) )

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
    annual[index]['top'] = pd.to_numeric(dic[index].set_index('years').top).groupby(level='years').mean()
    annual[index]['bottom'] = pd.to_numeric(dic[index].set_index('years').bottom).groupby(level='years').mean()
    monthly[index] = pd.DataFrame(pd.to_numeric(dic[index].set_index(['years','months']).lat).groupby(['years','months']).mean())
    monthly[index]['ua'] = pd.to_numeric(dic[index].set_index(['years','months']).ua).groupby(['years','months']).mean()
    monthly[index]['width'] = pd.to_numeric(dic[index].set_index(['years','months']).width).groupby(['years','months']).mean()
    monthly[index]['pr'] = pd.to_numeric(dic[index].set_index(['years','months']).pr).groupby(['years','months']).mean()
    monthly[index]['top'] = pd.to_numeric(dic[index].set_index(['years','months']).top).groupby(['years','months']).mean()
    monthly[index]['bottom'] = pd.to_numeric(dic[index].set_index(['years','months']).bottom).groupby(['years','months']).mean()
    winter[index] = pd.DataFrame(pd.to_numeric(dic[index].where(is_winter(dic[index].months)).dropna().set_index('years').lat).groupby(level='years').mean())
    winter[index]['ua'] = pd.to_numeric(dic[index].where(is_winter(dic[index].months)).dropna().set_index('years').ua).groupby(level='years').mean()
    winter[index]['width'] = pd.to_numeric(dic[index].where(is_winter(dic[index].months)).dropna().set_index('years').width).groupby(level='years').mean()
    winter[index]['pr'] = pd.to_numeric(dic[index].where(is_winter(dic[index].months)).dropna().set_index('years').pr).groupby(level='years').mean()
    winter[index]['top'] = pd.to_numeric(dic[index].where(is_winter(dic[index].months)).dropna().set_index('years').top).groupby(level='years').mean()
    winter[index]['bottom'] = pd.to_numeric(dic[index].where(is_winter(dic[index].months)).dropna().set_index('years').bottom).groupby(level='years').mean()
    seasonal[index] = pd.DataFrame(pd.to_numeric(dic[index].set_index(['years','seasons']).lat).groupby(level=['years','seasons']).mean())
    seasonal[index]['ua'] = pd.to_numeric(dic[index].set_index(['years','seasons']).ua).groupby(level=['years','seasons']).mean()
    seasonal[index]['width'] = pd.to_numeric(dic[index].set_index(['years','seasons']).width).groupby(level=['years','seasons']).mean()
    seasonal[index]['pr'] = pd.to_numeric(dic[index].set_index(['years','seasons']).pr).groupby(level=['years','seasons']).mean()
    seasonal[index]['top'] = pd.to_numeric(dic[index].set_index(['years','seasons']).top).groupby(level=['years','seasons']).mean()
    seasonal[index]['bottom'] = pd.to_numeric(dic[index].set_index(['years','seasons']).bottom).groupby(level=['years','seasons']).mean()



# mean var winter anoma Summ anom pr correlation
df = pd.DataFrame([],columns=["model" ,"latitude" ,"Velocity", "Width","Northern Extent","Southern Extent", "Lat std", "ua std", "width std" ,"Northern Extent std","Southern Extent std" ,"lat anom" , "ua anom" , "width anom","Northern Extent anom","Southern Extent anom","lat pr corr" ,"lat p-value", "ua pr corr" , "ua p-value", "width pr corr", "width p-value","Northern Extent corr","Northern Extent p-value","Southern Extent corr","Southern Extent p-values"])

for index in dic:
    df.loc[len(df)] = [
    index,
    round(np.mean(dic[index].lat),2),
    round(np.mean(dic[index].ua),2),
    round(np.mean(dic[index].width),2),
    round(np.mean(dic[index].top),2),
    round(np.mean(dic[index].bottom),2),
    round(np.std(dic[index].lat),2),
    round(np.std(dic[index].ua),2),
    round(np.std(dic[index].width),2),
    round(np.std(dic[index].top),2),
    round(np.std(dic[index].bottom),2),
    round(np.mean(np.mean(dic[index].lat) - winter[index].lat),2),
    round(np.mean(np.mean(dic[index].ua) - winter[index].ua),2),
    round(np.mean(np.mean(dic[index].width) - winter[index].width),2),
    round(np.mean(np.mean(dic[index].top) - winter[index].top),2),
    round(np.mean(np.mean(dic[index].bottom) - winter[index].bottom),2),
    round(scipy.stats.pearsonr(list(seasonal[index].lat),list(seasonal[index].pr))[0],2),
    format(scipy.stats.pearsonr(list(seasonal[index].lat),list(seasonal[index].pr))[1],'.2e'),
    round(scipy.stats.pearsonr(list(seasonal[index].ua),list(seasonal[index].pr))[0],2),
    format(scipy.stats.pearsonr(list(seasonal[index].ua),list(seasonal[index].pr))[1],'.2e'),
    round(scipy.stats.pearsonr(list(seasonal[index].width),list(seasonal[index].pr))[0],2),
    format(scipy.stats.pearsonr(list(seasonal[index].width),list(seasonal[index].pr))[1],'.2e'),
    format(scipy.stats.pearsonr(list(seasonal[index].top),list(seasonal[index].pr))[0],'.2e'),
    format(scipy.stats.pearsonr(list(seasonal[index].top),list(seasonal[index].pr))[1],'.2e'),
    format(scipy.stats.pearsonr(list(seasonal[index].bottom),list(seasonal[index].pr))[0],'.2e'),
    format(scipy.stats.pearsonr(list(seasonal[index].bottom),list(seasonal[index].pr))[1],'.2e'),
    ]


dfloat = df.iloc[:,:1].join(df.iloc[:,1:].astype('float'))

for colname in df.columns:      #only p-values in scientific
    if "p-value" in colname:
        dfloat[colname] = df[colname]

#df.set_value('model','NOAA','NOAA 20CR')

dfloat.to_csv('../../JET_STREAM_OUT/jet1D/results_table.csv',float_format='%.3f')
