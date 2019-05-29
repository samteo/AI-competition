# -*- coding: utf-8 -*-
"""
Created on Thu May 16 16:26:38 2019

@author: Big data
"""

import datetime
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, skew
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from scipy.special import boxcox1p
from sklearn.model_selection import cross_val_score

def AVM(y_test, y_pred):
    y_test, y_pred = np.array(y_test), np.array(y_pred)
    Z = np.where(abs((y_pred - y_test)/y_test) <= 0.1, 1, 0)
    hit_rate = Z.sum()/len(Z)
    MAPE = (abs((y_pred - y_test)/y_test)).sum()/len(Z)
    score = hit_rate*10000 + (1-MAPE)
    return print(score)

df_train_withparking = pd.read_csv('withparking.csv')
y = df_train_withparking.pop('total_price')
df_test_withparking = pd.read_csv("withparking_test.csv")

df=pd.concat((df_train_withparking,df_test_withparking),axis=0)

for i in df.columns:
    if "index" in i:
        df[i]=df[i].astype("category")
df["parking_way"]=df["parking_way"].astype("category")
df["city"]=df["city"].astype("category")
df["town"]=df["town"].astype("category")
df["village"]=df["village"].astype("category")

df['village_income_median']=df['village_income_median'].fillna(df['village_income_median'].median())


df = df.drop(["N_10000","I_index_5000","I_index_10000","building_id","village"],axis=1)

for i in df.columns:
    if df[i].dtypes == "int64" or df[i].dtypes == "float64":
        skewness1 = df[i].skew()
        if  skewness1 > 0.75:
            df[i] = np.log1p(df[i])
            skewness2 = df[i].skew()
            print(i,skewness1,skewness2)
        elif skewness1 < -0.75:
            df[i] = np.cbrt(df[i])
            skewness2 = df[i].skew()
            print(i,skewness1,skewness2)
            
df=pd.get_dummies(df)
X_train_withparking = df[:13935]
X_test_withparking = df[13935:]
X_train, X_test, y_train, y_test = train_test_split(X_train_withparking, y, test_size=0.33, random_state=42)
y_train = np.log(y_train)


print('start:',datetime.datetime.now())
model = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=6, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
model.fit(X_train, y_train)
#scores = cross_val_score(model, X_train, y_train, cv=5)
#scores
y_pred = model.predict(X_test)
y_pred = np.exp(y_pred)
AVM(y_test, y_pred)
y_predfinal = model.predict(X_test_withparking)
y_predfinal= np.exp(y_predfinal)
df_predfinal=pd.DataFrame()
df_predfinal["building_id"] =df_test_withparking["building_id"]
df_predfinal["total_price"]= y_predfinal

df_predfinal.to_csv("withparking_result.csv",index=False)

print('end:',datetime.datetime.now())
