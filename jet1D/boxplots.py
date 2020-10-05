import numpy as np
import pandas as pd
import xarray as xr
import glob
import math
import statistics
import pickle
import matplotlib.pyplot as plt
import seaborn as sns


extended = pickle.load( open( "EXTENDED.p", "rb" ) )
seasonal = pickle.load( open( "SEASONAL.p", "rb" ) )
annual = pickle.load( open( "ANNUAL.p", "rb" ) )

seasonality = {} #anomaly
for index in seasonal:
    seasonality[index] = np.mean(annual[index].lat) - seasonal[index]['Winter'].lat #if annual is -30 and winter is -25 = -5 or 5 deg north

seasonality = {} #oscilation
for index in extended:
    rolling = extended[index]['Summer'].rolling(2).mean().dropna().reset_index()  #dropna results in neighbouring summers being considered
    seasonality[index] = rolling.lat - extended[index]['Winter'].lat[:-1]


#############################################################################
#plotting
#####################################################################

data = list(seasonality.values())
fig, ax1 = plt.subplots(figsize=(16, 9))
fig.canvas.set_window_title('CMIP5 Jet Seasonality')
fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

bp = ax1.boxplot(data, notch=False, sym='+', vert=1)#, whis=[1,99]) #currently set as 1.5*IQR
plt.setp(bp['boxes'], color='black')
plt.setp(bp['whiskers'], color='black')
plt.setp(bp['fliers'], color='red', marker='.')

# Add a horizontal grid to the plot, but make it very light in color
# so we can use it for reading data values but not be distracting
ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
               alpha=0.7)

# Hide these grid behind plot objects
ax1.set_axisbelow(True)
ax1.set_title('Seasonal Jet Anomaly')
ax1.set_xlabel('Model')
ax1.set_ylabel('Latitude Anomaly')

# Now fill the boxes with desired colors
box_colors = ['darkkhaki', 'royalblue']
num_boxes = len(data)
medians = np.empty(num_boxes)
for i in range(num_boxes):
    box = bp['boxes'][i]
    boxX = []
    boxY = []
    for j in range(5):
        boxX.append(box.get_xdata()[j])
        boxY.append(box.get_ydata()[j])
    box_coords = np.column_stack([boxX, boxY])
    # Alternate between Dark Khaki and Royal Blue
    #ax1.add_patch(Polygon(box_coords, facecolor=box_colors[i % 2]))
    # Now draw the median lines back over what we just filled in
    med = bp['medians'][i]
    medianX = []
    medianY = []
    for j in range(2):
        medianX.append(med.get_xdata()[j])
        medianY.append(med.get_ydata()[j])
        ax1.plot(medianX, medianY, 'k')
    medians[i] = medianY[0]
    # Finally, overplot the sample averages, with horizontal alignment
    # in the center of each box
    ax1.plot(np.average(med.get_xdata()), np.average(data[i]),
             color='w', marker='*', markeredgecolor='k')

# Set the axes ranges and axes labels
#ax1.set_xlim(0.5, num_boxes + 0.5)
top = 15
bottom = -15
ax1.set_ylim(bottom, top)
models = list(seasonality.keys())
ax1.set_xticklabels(models,
                    rotation=90, fontsize=8)

# Due to the Y-axis scale being different across samples, it can be
# hard to compare differences in medians across the samples. Add upper
# X-axis tick labels with the sample medians to aid in comparison
# (just use two decimal places of precision)
pos = np.arange(num_boxes) + 1
upper_labels = [str(np.round(s, 2)) for s in medians]
weights = ['bold', 'semibold']
for tick, label in zip(range(num_boxes), ax1.get_xticklabels()):
    if float(upper_labels[tick]) > 0:
        conditional_colour='red'
    #elif float(upper_labels[tick]) > -1 and float(upper_labels[tick]) < 0:
    #    conditional_colour = 'red'
    else:
        conditional_colour='green'
    ax1.text(pos[tick], .05, upper_labels[tick],
             transform=ax1.get_xaxis_transform(),
             horizontalalignment='center', size='small',
             color=conditional_colour)


plt.savefig('CMIP_Winter_box_anomally_jja.jpeg',dpi=95)
#plt.show()
###########################################################
summer={}
for index in seasonal:
    summer[index] = seasonal[index]['Summer'].lat

data_a = list(summer.values()) #summer

winter={}
for index in seasonal:
    winter[index] = seasonal[index]['Winter'].lat

data_b = list(winter.values()) #winter
ticks = list(summer.keys())

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


plt.figure(figsize=(16,9))
sum = plt.boxplot(data_a, positions=np.array(range(len(data_a)))*2.0-0.4, sym='', widths=0.6)
win = plt.boxplot(data_b, positions=np.array(range(len(data_b)))*2.0+0.4, sym='', widths=0.6)
set_box_color(sum, 'red') # colors are from http://colorbrewer2.org/
set_box_color(win, 'blue')
plt.plot([], c='red', label='Summer [DJF]') # draw temporary red and blue lines and use them to create a legend
plt.plot([], c='blue', label='Winter [JJA]')
plt.legend()
plt.xticks(range(0, len(ticks) * 2, 2), ticks,rotation=90)
plt.tight_layout()
plt.title('CMIP5 Seasonal Jet Stream Analysis')

plt.subplots_adjust(top=0.93)
plt.savefig('CMIP_seasonal_box_compare.jpeg',dpi=95)

plt.show()
