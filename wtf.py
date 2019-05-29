# -*- coding: utf-8 -*-
"""
Created on Sat May 25 18:07:44 2019

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


df_train_noparking = pd.read_csv('noparking1.csv')

'''
for i in df_train_noparking["building_use"].unique():
    mean = df_train_noparking["total_price"][df_train_noparking["building_use"]==i].mean()
    std = df_train_noparking["total_price"][df_train_noparking["building_use"]==i].std()
    print(i, "mean:{}".format(mean),
          "std:{}".format(std),
          "cv:{}".format(std/mean))'''

y = df_train_noparking.pop('total_price')
df_test_noparking = pd.read_csv("noparking1_test.csv")
#aa=df_train_noparking[df_train_noparking["building_use"]==2]
#bb=df_test_noparking[df_test_noparking["building_use"]!=1]
#y=aa.pop('total_price')

df=pd.concat((df_train_noparking,df_test_noparking),axis=0)


for i in df.columns:
    if "index" in i:
        df[i]=df[i].astype("category")
df["building_use"]=df["building_use"].astype("category")
df["city"]=df["city"].astype("category")
#df["town"]=df["town"].astype("category")
df["building_material"]=df["building_material"].astype("category")
df["house_age"]=(df["txn_dt"].sub(df["building_complete_dt"]))/365  
df["house_age_square"] = df["house_age"]**2
df["total_area"]=df['land_area'].add(df["building_area"])


median = df['village_income_median'].median()
df['village_income_median']=df['village_income_median'].fillna(df['village_income_median'].median())


df = df.drop(["N_10000","I_index_5000","I_index_10000","II_index_5000","II_index_10000","III_index_5000","III_index_10000","IV_index_10000",
              "V_index_5000","V_index_10000","VI_index_5000","VI_index_10000","VII_index_5000","VII_index_10000","VIII_index_5000","VIII_index_10000",
              "IX_index_5000","IX_index_10000","X_index_5000","X_index_10000","XI_index_10000","XII_index_5000","XII_index_10000","XIV_index_5000",
              "XIV_index_10000","building_type","town",
              "VI_10","building_id","parking_way","parking_price","txn_floor","village"],axis=1)
for i in df.columns:
    if df[i].dtypes == "int64" or df[i].dtypes == "float64" :
        skewness1 = df[i].skew()
        if  skewness1 > 0.75:
            df[i] = boxcox1p(df[i],0.15)
            skewness2 = df[i].skew()
            print(i,skewness1,skewness2)
        elif skewness1 < -0.75:
            df[i] = np.cbrt(df[i])
            skewness2 = df[i].skew()
            print(i,skewness1,skewness2)

df=pd.get_dummies(df)
X_train_noparking = df[:15810]
X_test_noparking = df[15810:]
X_train, X_test, y_train, y_test = train_test_split(X_train_noparking, y, test_size=0.33, random_state=42)
X_train = X_train[y_train>250000]
y_train = y_train[y_train>250000]
y_train = np.log(y_train)


print('start:',datetime.datetime.now())
model = xgb.XGBRegressor(colsample_bytree=0.4, gamma=0.01, 
                             learning_rate=0.01, max_depth=9, 
                             min_child_weight=0.2, n_estimators=4000,
                             reg_alpha=0.1, reg_lambda=0.1,
                             subsample=0.3, silent=1,nthread =8,
                             random_state =7)
model.fit(X_train, y_train)
#scores = cross_val_score(model, X_train, y_train, cv=5)
#print(scores)
y_pred = model.predict(X_test)
y_pred = np.exp(y_pred)
AVM(y_test, y_pred)
