import codecs
import json

import pandas as pd
import xgboost
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from termcolor import colored as cl # text customization


df = pd.read_csv("skane_house_price_clean.csv")
df['rent'].replace(' ', np.nan, inplace=True)
df.dropna(subset=['rent'], inplace=True)
df['rent'] = df['rent'].astype(float)

print(cl(df.dtypes, attrs = ['bold']))

sb.heatmap(df.corr(), annot = True, cmap = 'magma')

plt.savefig('plots/heatmap.png')
plt.show()


def scatter_df(y_var):
    #scatter_df = df.drop(y_var, axis=1)
    i = df.columns

    plot1 = sb.scatterplot(i[2], y_var, data=df, color='orange', edgecolor='b', s=150)
    plt.title('{} / Sale Price'.format(i[2]), fontsize=16)
    plt.xlabel('{}'.format(i[2]), fontsize=14)
    plt.ylabel('{}'.format(i[1]), fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('plots/scatter1.png')
    plt.show()

    plot1 = sb.scatterplot(i[3], y_var, data=df, color='orange', edgecolor='b', s=150)
    plt.title('{} / Sale Price'.format(i[3]), fontsize=16)
    plt.xlabel('{}'.format(i[3]), fontsize=14)
    plt.ylabel('{}'.format(i[1]), fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('plots/scatter2.png')
    plt.show()

    plot2 = sb.scatterplot(i[4], y_var, data=df, color='yellow', edgecolor='b', s=150)
    plt.title('{} / Sale Price'.format(i[4]), fontsize=16)
    plt.xlabel('{}'.format(i[4]), fontsize=14)
    plt.ylabel('{}'.format(i[1]), fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('plots/scatter3.png')
    plt.show()

    plot2 = sb.scatterplot(i[5], y_var, data=df, color='yellow', edgecolor='b', s=150)
    plt.title('{} / Sale Price'.format(i[5]), fontsize=16)
    plt.xlabel('{}'.format(i[5]), fontsize=14)
    plt.ylabel('{}'.format(i[1]), fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('plots/scatter4.png')
    plt.show()

scatter_df('price')
