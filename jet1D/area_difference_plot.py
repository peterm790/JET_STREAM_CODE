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


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


for index in dic:
    fig, ax1 = plt.subplots(figsize=(10,6))
    fig.subplots_adjust(right=0.75)
    ax = sns.kdeplot(list(dic[index].lat),lw=2.5,color='r', shade=True, vertical=False, label='Model Latitude',ax=ax1)
    ax = sns.kdeplot(list(dic['NOAA'].lat),lw=2,color='r',shade=True,linestyle = ':', vertical=False,label='Reananalysis Latitude', ax = ax1)
    l1 = ax.lines[0]
    l2 = ax.lines[1]
    x1, y1 = l1.get_xydata().T
    x2, y2 = l2.get_xydata().T
    xmin = max(x1.min(), x2.min())
    xmax = min(x1.max(), x2.max())
    x = np.linspace(xmin, xmax, 100)
    y1 = np.interp(x, x1, y1)
    y2 = np.interp(x, x2, y2)
    y = np.minimum(y1, y2)
    ax.fill_between(x, y, color="white", alpha=1)
    tkw = dict(size=4, width=1.5)
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend()
    plt.title(str(index)+' Jet Latitude Area difference', fontsize=16, ha='center')
    plt.savefig('../../JET_STREAM_OUT/jet1D/area_difference/'+str(index)+'_lat_area_difference.jpeg',dpi=95)
    plt.close()
