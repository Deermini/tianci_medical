# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 17:08:02 2018

King
"""

import time
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from dateutil.parser import parse
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error



data_path = 'E:/python/tianchi/'

train = pd.read_csv(data_path+'d_train_2018.csv',encoding='gb2312')
test = pd.read_csv(data_path+'d_test_A_2018.csv',encoding='gb2312')

def make_feat(train,test):
    train_id = train.id.values.copy()
    test_id = test.id.values.copy()
    data = pd.concat([train,test])

    data['性别'] = data['性别'].map({'男':1,'女':0})
    data['体检日期'] = (pd.to_datetime(data['体检日期']) - parse('2017-10-09')).dt.days
 
    data.fillna(data.median(axis=0),inplace=True)

    train_feat = data[data.id.isin(train_id)]
    test_feat = data[data.id.isin(test_id)]

    return train_feat,test_feat


    
df_train = train.iloc[:1363,:37]
df_train1 = train.iloc[1363:,:37]    
df_test = test.iloc[:168,:37]
df_test_y = test.iloc[:168,37] 
df_test1 = test.iloc[168:,:37]  
df_con_train = pd.concat([df_train,df_test],axis=0)    
df_con_test = pd.concat([df_train1,df_test1],axis=0)    
 
train_feat,test_feat = make_feat(df_con_train,df_con_test)

predictors = [f for f in test_feat.columns if f not in ['乙肝e抗体']]
   
    
    
def evalerror(pred, df):
    label = df.get_label().values.copy()
    score = mean_squared_error(label,pred)*0.5
    return ('0.5mse',score,False)

print('开始训练...')
params = {
    'learning_rate': 0.01,
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mse',
    'sub_feature': 0.7,
    'num_leaves': 60,
    'colsample_bytree': 0.7,
    'feature_fraction': 0.7,
    'min_data': 100,
    'min_hessian': 1,
    'verbose': -1,
}

print('开始CV 5折训练...')
t0 = time.time()
train_preds = np.zeros(train_feat.shape[0])
test_preds = np.zeros((test_feat.shape[0], 5))
kf = KFold(len(train_feat), n_folds = 5, shuffle=True, random_state=520)
for i, (train_index, test_index) in enumerate(kf):
    print('第{}次训练...'.format(i))
    train_feat1 = train_feat.iloc[train_index]
    train_feat2 = train_feat.iloc[test_index]
    lgb_train1 = lgb.Dataset(train_feat1[predictors], train_feat1['乙肝e抗体'],categorical_feature=['性别'])
    lgb_train2 = lgb.Dataset(train_feat2[predictors], train_feat2['乙肝e抗体'])
    gbm = lgb.train(params,
                    lgb_train1,
                    num_boost_round=3000,
                    valid_sets=lgb_train2,
                    verbose_eval=100,
                    feval=evalerror,
                    early_stopping_rounds=100)
    feat_imp = pd.Series(gbm.feature_importance(), index=predictors).sort_values(ascending=False)
    train_preds[test_index] += gbm.predict(train_feat2[predictors])
    test_preds[:,i] = gbm.predict(test_feat[predictors])
print('线下得分：    {}'.format(mean_squared_error(train_feat['乙肝e抗体'],train_preds)*0.5))
print('CV训练用时{}秒'.format(time.time() - t0))

submission = pd.DataFrame({'pred':test_preds.mean(axis=1)})
submission.to_csv(r'sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),header=None,
                  index=False, float_format='%.4f')    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    