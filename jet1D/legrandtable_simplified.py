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

dic = pickle.load( open( "daily_jettrack.p", "rb" ) )

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



# mean var winter anoma Summ anom pr correlation
df = pd.DataFrame([],columns=["model" ,"latitude" ,"Velocity", "Width", "Lat std", "ua std", "width std" , "lat anom" , "ua anom" , "width anom", "lat pr corr" ,"lat p-value", "ua pr corr" , "ua p-value", "width pr corr", "width p-value"])

for index in dic:
    df.loc[len(df)] = [
    index,
    round(np.mean(dic[index].lat),2),
    round(np.mean(dic[index].ua),2),
    round(np.mean(dic[index].width),2),
    round(np.std(dic[index].lat),2),
    round(np.std(dic[index].ua),2),
    round(np.std(dic[index].width),2),
    round(np.mean(np.mean(dic[index].lat) - winter[index].lat),2),
    round(np.mean(np.mean(dic[index].ua) - winter[index].ua),2),
    round(np.mean(np.mean(dic[index].width) - winter[index].width),2),
    round(scipy.stats.pearsonr(list(seasonal[index].lat),list(seasonal[index].pr))[0],2),
    format(scipy.stats.pearsonr(list(seasonal[index].lat),list(seasonal[index].pr))[1],'.2e'),
    round(scipy.stats.pearsonr(list(seasonal[index].ua),list(seasonal[index].pr))[0],2),
    format(scipy.stats.pearsonr(list(seasonal[index].ua),list(seasonal[index].pr))[1],'.2e'),
    round(scipy.stats.pearsonr(list(seasonal[index].width),list(seasonal[index].pr))[0],2),
    format(scipy.stats.pearsonr(list(seasonal[index].width),list(seasonal[index].pr))[1],'.2e')
    ]


#df.set_value('model','NOAA','NOAA 20CR')

df.to_csv('legranddtable_simplified.csv')
