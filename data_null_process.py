# -*- coding: utf-8 -*-
'''

Time : 2017\12\29 0029 16:55
King

'''



import pandas as pd
from sklearn.preprocessing import Imputer
from scipy.interpolate import lagrange

df = pd.read_csv('data_train.csv')
print(df.shape)
print(df.dtypes)
'''
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(df)
a = imp.transform(df)
print(pd.DataFrame(a).describe())
pd.DataFrame(a).to_csv('2.csv')

'''

def ploy(s, n, k=30):
    y = s[list(range(n-k, n))+list(range(n+1, n+1+k))]#取数使用缺失值前后30个未缺失的数据进行建模
    y = y[y.notnull()]
    return lagrange(y.index, list(y))(n)

for i in df.columns:
    print(i)
    for j in range(len(df)):
        if(df[i].isnull())[j]:
            df[i][j] = ploy(df[i], j)
#df.to_csv('1.csv')













