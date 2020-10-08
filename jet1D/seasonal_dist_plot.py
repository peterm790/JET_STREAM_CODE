import numpy as np
import pandas as pd
import xarray as xr
import glob
import math
import statistics
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

dic = pickle.load( open( "../../JET_STREAM_OUT/jet1D/jettrack.p", "rb" ) )

def is_winter(month):
    return (month == 6) | (month == 7) | (month == 8)

def is_summer(month):
    return (month == 12) | (month == 1) | (month == 2)

def is_autumn(month):
    return (month == 3) | (month == 4) | (month == 5)

def is_spring(month):
    return (month == 9) | (month == 10) | (month == 11)


label = {'lat':'latitude',
'width':'width',
'ua':'velocity',
'top':'Northern Extent',
'bottom':'Southern Extent'}



for metric in ['lat','width','ua','top','bottom']:
    winter = pd.to_numeric(dic['NOAA'].where(is_winter(dic['NOAA'].months)).dropna().set_index('years')[metric])
    summer = pd.to_numeric(dic['NOAA'].where(is_summer(dic['NOAA'].months)).dropna().set_index('years')[metric])
    spring = pd.to_numeric(dic['NOAA'].where(is_spring(dic['NOAA'].months)).dropna().set_index('years')[metric])
    autumn = pd.to_numeric(dic['NOAA'].where(is_autumn(dic['NOAA'].months)).dropna().set_index('years')[metric])
    fig, ax1 = plt.subplots(figsize=(10,6))
    sns.kdeplot(list(winter),lw=3.5,color='b', vertical=True, label='Winter Width',ax=ax1)
    sns.kdeplot(list(summer),lw=3.5,color='r', vertical=True, label='Summer Width',ax=ax1)
    sns.kdeplot(list(spring),lw=3.5,color='g', vertical=True, label='spring Width',ax=ax1)
    sns.kdeplot(list(autumn),lw=3.5,color='y', vertical=True, label='Autumn Width',ax=ax1)
    ax1.set_xlabel("Kernel Density Estimate")
    ax1.set_ylabel(str(metric))
    plt.title('NOAA Jet '+label[metric]+' Seasonal Distribution Shift', fontsize=16, ha='center')
    plt.savefig('../../JET_STREAM_OUT/jet1D/seasonal_shift/png/'+str(metric)+'_seasonality.png',dpi=100)
    plt.savefig('../../JET_STREAM_OUT/jet1D/seasonal_shift/pdf/'+str(metric)+'_seasonality.pdf')
    plt.savefig('../../JET_STREAM_OUT/jet1D/seasonal_shift/svg/'+str(metric)+'_seasonality.svg',format='svg', dpi=1200)
    plt.close()
