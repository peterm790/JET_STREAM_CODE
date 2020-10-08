import numpy as np
import pandas as pd
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
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.18))
    make_patch_spines_invisible(ax3)
    ax3.spines["right"].set_visible(True)
    sns.kdeplot(list(dic[index].lat),lw=2.5,color='g', vertical=True, label='Model Latitude',ax=ax1)
    for index2 in dic:
        sns.kdeplot(list(dic[index2].lat),lw=.2,color='g', vertical=True ,ax=ax1)
    sns.kdeplot(list(dic[index].ua),color='r', vertical=True,lw=2.5, label='Model Velocity',ax=ax2)
    for index2 in dic:
        sns.kdeplot(list(dic[index2].ua),lw=.2,color='r', vertical=True ,ax=ax2)
    sns.kdeplot(list(dic[index].width),color='b', vertical=True,lw=2.5, label='Model Width',ax=ax3)
    for index2 in dic:
        sns.kdeplot(list(dic[index2].width),lw=.2,color='b', vertical=True ,ax=ax3)
    sns.kdeplot(list(dic['NOAA'].lat),lw=2,color='g',linestyle = ':', vertical=True,label='Reananalysis Latitude', ax = ax1)
    sns.kdeplot(list(dic['NOAA'].ua),color='r',linestyle=':', vertical=True,label='Reananalysis Velocity', lw=2,ax=ax2)
    sns.kdeplot(list(dic['NOAA'].width),color='b',linestyle=':', vertical=True,lw=2, label = 'Reananalysis Width', ax=ax3)
    ax1.set_xlabel("Kernel Density Estimate")
    ax1.set_ylabel("Latitude")
    ax2.set_ylabel("Velocity")
    ax3.set_ylabel("Width")
    ax1.yaxis.label.set_color('g')
    ax2.yaxis.label.set_color('r')
    ax3.yaxis.label.set_color('b')
    tkw = dict(size=4, width=1.5)
    ax1.tick_params(axis='y', colors='g', **tkw)
    ax2.tick_params(axis='y', colors='r', **tkw)
    ax3.tick_params(axis='y', colors='b', **tkw)
    ax1.tick_params(axis='x', **tkw)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax1.get_legend().remove()
    ax2.get_legend().remove()
    ax3.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc=0)
    plt.title(str(index)+' Jet Analyses', fontsize=16, ha='center')
    plt.savefig('../../JET_STREAM_OUT/jet1D/busyplot/pdf/busyplot_'+str(index)+'.pdf')
    plt.savefig('../../JET_STREAM_OUT/jet1D/busyplot/svg/busyplot_'+str(index)+'.svg', format='svg', dpi=1200)
    plt.savefig('../../JET_STREAM_OUT/jet1D/busyplot/png/busyplot_'+str(index)+'.png', format='png', dpi=1200)
    plt.close()
