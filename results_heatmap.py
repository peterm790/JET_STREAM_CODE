### import the things
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def getnormalizerd(df):
    normalized_df=(df-df.min())/(df.max()-df.min())
    return normalized_df

#from uwind
daily1d = getnormalizerd(pd.read_csv('../uwind/scores_daily.csv',index_col='Model').drop('EC-EARTH')).add_suffix('_daily')
seasonal1d = getnormalizerd(pd.read_csv('../uwind/scores_seasonal.csv',index_col='Model').drop('EC-EARTH')).add_suffix('_seasonal')
decomp1d = getnormalizerd(pd.read_csv('../uwind/scores_decomposed.csv',index_col='Model').drop('EC-EARTH')).add_suffix('_decomposed')

#from visualizeR
daily3d = getnormalizerd(pd.read_csv('../visualizeR/scores_daily_3d.csv',index_col='Model').iloc[:,1:]).add_suffix('_daily_3d')
seasonal3d = getnormalizerd(pd.read_csv('../visualizeR/scores_seasonal_3d.csv',index_col='Model').iloc[:,1:]).add_suffix('_seasonal_3d')
decomp3d = getnormalizerd(pd.read_csv('../visualizeR/scores_seasonal_decomposed.csv',index_col='Model').iloc[:,1:]).add_suffix('_decomposed_3d')


mean2d = getnormalizerd(pd.read_csv('../visualizeR/scores_mean.csv',index_col='Model').iloc[:,1:]).add_suffix('_area_mean')
mean2d_decomp =getnormalizerd(pd.read_csv('../visualizeR/scores_mean_decomp.csv',index_col='Model').iloc[:,1:]).add_suffix('_area_mean_decomp')

dfs = [daily1d,seasonal1d,decomp1d,daily3d,seasonal3d,decomp3d,mean2d,mean2d_decomp]
df = pd.DataFrame().join(dfs, how="outer")


#df.pop('ua_daily')
#df.pop('ua_seasonal')
#df.pop('ua_decomposed')

#df['mean_score'] = getnormalizerd(df.iloc[:,:-1].mean(axis=1))

df = df.sort_index() #impartial I guess
df = df.sort_values('score_decomposed_3d')
fulldf = df

df = fulldf[['ua_daily','ua_seasonal','ua_decomposed','width_daily','width_seasonal','width_decomposed','lat_daily','lat_seasonal','lat_decomposed','score_daily_3d','score_seasonal_3d','score_decomposed_3d','score_area_mean','score_area_mean_decomp']]



import seaborn as sns
sns.set()

# Draw a heatmap with the numeric values in each cell
f, ax = plt.subplots(figsize=(16,12))
map = sns.heatmap(df.round(3),cmap=sns.color_palette("RdPu", 5), annot=True, linewidths=.5, ax=ax, xticklabels=1, yticklabels=1)
map.set_xticklabels(['Velocity Daily','Velocity Seasonal','Velocity Decomposed','Width Daily','Width Seasonal','Width Decomposed','Latitude Daily','Latitude Seasonal','Latitude Decomposed','2D Daily','2D Seasonal','2D Decomposed','Mean WS','Seasonal Mean WS'],rotation=45,horizontalalignment='right')
plt.title('Jet Stream Final Scores - Scaled')
plt.savefig('Scores_heatmap.pdf')
plt.show()

corrs = df.corr()
f, ax = plt.subplots(figsize=(16,12))
map = sns.heatmap(corrs.round(3), cmap = sns.color_palette("BuGn",5),vmin=0,annot=True,linewidths=.5, ax=ax, xticklabels=1, yticklabels=1)
map.set_xticklabels(['Velocity Daily','Velocity Seasonal','Velocity Decomposed','Width Daily','Width Seasonal','Width Decomposed','Latitude Daily','Latitude Seasonal','Latitude Decomposed','2D Daily','2D Seasonal','2D Decomposed','Mean WS','Seasonal Mean WS'],rotation=45,horizontalalignment='right')
map.set_yticklabels(['Velocity Daily','Velocity Seasonal','Velocity Decomposed','Width Daily','Width Seasonal','Width Decomposed','Latitude Daily','Latitude Seasonal','Latitude Decomposed','2D Daily','2D Seasonal','2D Decomposed','Mean WS','Seasonal Mean WS'],rotation=0,horizontalalignment='right')
plt.title('Jet Stream Score Correlations')
plt.savefig('Scores_Correlation_map.pdf')
plt.show()



df = fulldf[['ua_decomposed','width_decomposed','lat_decomposed','score_decomposed_3d','score_area_mean_decomp']]

f, ax = plt.subplots(figsize=(16,12))
map = sns.heatmap(df.round(3),cmap=sns.color_palette("RdPu", 4), annot=True, linewidths=.5, ax=ax, xticklabels=1, yticklabels=1)
map.set_xticklabels(['Velocity','Width','Latitude','2D','Mean WS'],rotation=45,horizontalalignment='right')
plt.title('Jet Stream Final Scores - Scaled')
plt.savefig('Scores_heatmap_decomp.pdf')
plt.show()


corrs = df.corr()
f, ax = plt.subplots(figsize=(16,12))
map = sns.heatmap(corrs.round(3), cmap = sns.color_palette("BuGn",5),vmin=0,annot=True,linewidths=.5, ax=ax, xticklabels=1, yticklabels=1)
map.set_xticklabels(['Velocity','Width','Latitude','2D','Mean WS'],rotation=45,horizontalalignment='right')
map.set_yticklabels(['Velocity','Width','Latitude','2D','Mean WS'],rotation=0,horizontalalignment='right')
plt.title('Jet Stream Score Correlations')
plt.savefig('Scores_Correlation_map_decomp.pdf')
plt.show()


















for index in df.score_decomposed_3d.index[3:]:
    print('\subcaptionbox{'+str(index)+'\label{CMIP5 Fig2}}{\includegraphics[width = 3.3cm,trim={1.5cm 1.5cm 1.5cm 1.5cm},clip]{visualizeR/correlationplots_2deg/'+str(index)+'_Jet_correlation.pdf}}')
