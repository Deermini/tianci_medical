# -*- coding: utf-8 -*-
'''

Time : 2017\12\28 0028 16:51
King

'''

import numpy as np
import pandas as pd
from matplotlib import pyplot
#from pandas import  plotting.scatter_matrix
from pandas.tools.plotting import scatter_matrix


df = pd.read_csv('TianChi\d_train.csv',encoding='cp936')
print(df.shape)
print(df.dtypes)


describe = df.describe()
print("describe:",describe)
df.fillna(-1, inplace=True)
cor = df.corr(method='pearson')
# pd.DataFrame(describe).to_csv('describe.csv')
# pd.DataFrame(cor).to_csv('cor.csv')


#数据分布
# df.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1)
# df.plot(kind='density', subplots=True, layout=(7, 7), sharex=False, fontsize=1)
# df.plot(kind='box', subplots=True, layout=(7, 7), sharex=False, sharey=False, fontsize=8)
# #scatter_matrix(df)
# pyplot.show()
#

names = [column for column in df]
# names =df.columns.values
print(names)
# 相关矩阵图
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(df.corr(), vmin=-1, vmax=1, interpolation='none')
fig.colorbar(cax)
ticks = np.arange(0, 41, 1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
pyplot.xticks(rotation='vertical')
pyplot.show()

df['sam_null'] = (df < 0).sum(axis=1)
#print("df['sam_null']:",df['sam_null'])
feature_null = (df < 0).sum(axis=0)
#df['sam_null'].to_csv('na.csv')

#缺失值观察
#pyplot.scatter(df['id'], df['sam_null'], c='r')

pyplot.plot(range(len(feature_null)), feature_null, '--r*')
pyplot.xlim((0, 50))
pyplot.ylim((0, 5000))
my_x_ticks = np.arange(0, 50, 1)
my_y_ticks = np.arange(0, 5000, 100)
pyplot.xticks(my_x_ticks)
pyplot.yticks(my_y_ticks)
pyplot.show()






















