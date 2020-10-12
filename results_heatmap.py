import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()


def getnormalizerd(df):
    normalized_df=(df-df.min())/(df.max()-df.min())
    return normalized_df

#from uwind
scores_1d = getnormalizerd(pd.read_csv('../JET_STREAM_OUT/jet1D/scores.csv',index_col='Model').drop('EC-EARTH')).add_suffix('')

#from visualizeR
scores_2d = getnormalizerd(pd.read_csv('../JET_STREAM_OUT/jet2D/scores.csv',index_col='Model').iloc[:,1:]).add_suffix('_2d')



dfs = [scores_1d,scores_2d]
df = pd.DataFrame().join(dfs, how="outer")


df = df.sort_values('score_2d')


f, ax = plt.subplots(figsize=(16,12))
map = sns.heatmap(df.round(3),cmap=sns.color_palette("RdPu", 4), annot=True, linewidths=.5, ax=ax, xticklabels=1, yticklabels=1)
map.set_xticklabels(['Velocity','Width','Latitude','2-Dimensional'],rotation=45,horizontalalignment='right')
plt.title('Jet Stream Scores')
plt.savefig('../JET_STREAM_OUT/Scores_heatmap.pdf')
plt.close()


corrs = df.corr()
f, ax = plt.subplots(figsize=(16,12))
map = sns.heatmap(corrs.round(3), cmap = sns.color_palette("BuGn",5),vmin=0,annot=True,linewidths=.5, ax=ax, xticklabels=1, yticklabels=1)
map.set_xticklabels(['Velocity','Width','Latitude','2D'],rotation=45,horizontalalignment='right')
map.set_yticklabels(['Velocity','Width','Latitude','2D'],rotation=0,horizontalalignment='right')
plt.title('Jet Stream Score Correlations')
plt.savefig('../JET_STREAM_OUT/Scores_Correlation_heatmap.pdf')
plt.close()
