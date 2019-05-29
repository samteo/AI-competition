# -*- coding: utf-8 -*-
"""
Created on Fri May 24 14:50:39 2019
testcombine
@author: Big data
"""
#df["txn_floor"].where(df["txn_floor"].isna(),df["total_floor"])

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
from sklearn.model_selection import GridSearchCV

def AVM(y_test, y_pred):
    y_test, y_pred = np.array(y_test), np.array(y_pred)
    Z = np.where(abs((y_pred - y_test)/y_test) <= 0.1, 1, 0)
    hit_rate = Z.sum()/len(Z)
    MAPE = (abs((y_pred - y_test)/y_test)).sum()/len(Z)
    score = hit_rate*10000 + (1-MAPE)
    return print(score)

df_train_noparking = pd.read_csv('testcombine.csv')
y = df_train_noparking.pop('total_price')
df_test_noparking = pd.read_csv("testcombine_test.csv")
df=pd.concat((df_train_noparking,df_test_noparking),axis=0)

for i in df.columns:
    if "index" in i:
        df[i]=df[i].astype("category")
#df["txn_floor"][df["txn_floor"].isna()]= df["total_floor"][df["txn_floor"].isna()].to_numpy() #na與沒na都沒差,沒有totalfloor結果更好
df['lat']=df['lat'].sub(-42)        
#df["building_use"]=df["building_use"].astype("category")
#df["city"]=df["city"].astype("category")
#df["town"]=df["town"].astype("category")
#df["building_material"]=df["building_material"].astype("category")
df["house_age"]=(df["txn_dt"].sub(df["building_complete_dt"]))/365  
df["house_age_square"] = df["house_age"]**2
df["total_area"]=df['land_area'].add(df["building_area"])      
median = df['village_income_median'].median()
df['village_income_median']=df['village_income_median'].fillna(df['village_income_median'].median())
#有值"XIII_index_5000"]"XI_index_5000"]
#"V_index_5000","V_index_10000",
#df.drop(["N_10000","I_index_5000","I_index_10000","II_index_5000","II_index_10000","VI_10","building_id","parking_way","parking_price"],axis=1)
df = df.drop(["N_10000","I_index_5000","I_index_10000","II_index_5000","II_index_10000","III_index_5000","III_index_10000","IV_index_10000",
              "V_index_5000","V_index_10000","VI_index_5000","VI_index_10000","VII_index_5000","VII_index_10000","VIII_index_5000","VIII_index_10000",
              "IX_index_5000","IX_index_10000","X_index_5000","X_index_10000","XI_index_10000","XII_index_5000","XII_index_10000","XIV_index_5000",
              "XIV_index_10000",
              "VI_10","building_id","parking_way","parking_price"],axis=1)

for i in df.columns:
    if df[i].dtypes == "int64" or df[i].dtypes == "float64":
        skewness1 = df[i].skew()
        if  skewness1 > 0.75:
            df[i] = np.log1p(df[i])
            skewness2 = df[i].skew()
            print(i,skewness1,skewness2)
        elif skewness1 < -0.75:
            df[i] = np.power(df[i],2)
            skewness2 = df[i].skew()
            print(i,skewness1,skewness2)
            
 
df=pd.get_dummies(df)
X_train_noparking = df[:46065]
X_test_noparking = df[46065:]
X_train, X_test, y_train, y_test = train_test_split(X_train_noparking, y, test_size=0.33, random_state=42)
#X_train = X_train[y_train<30000000]
#y_train = y_train[y_train<30000000]
y_train = np.log(y_train)

print('start:',datetime.datetime.now())
#還沒調
'''model = xgb.XGBRegressor(colsample_bytree=0.5, gamma=0.0468, 
                             learning_rate=0.035, max_depth=13, 
                             min_child_weight=1.7817, n_estimators=2500,
                             reg_alpha=0.4640, reg_lambda=0.8,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

param_dist = {
        'n_estimators':range(500,1000,200),
        'max_depth':range(8,25,1),
        'subsample':np.linspace(0.5,0.9,20),
        'colsample_bytree':np.linspace(0.4,0.9,10),
        'min_child_weight':np.linspace(1,3,20)
        }
#調第一次
model = xgb.XGBRegressor(colsample_bytree=0.70, gamma=0.023, 
                             learning_rate=0.02, max_depth=9, 
                             min_child_weight=1.222, n_estimators=3500,
                             reg_alpha=0.6, reg_lambda=1,
                             subsample=0.625, silent=1,
                             random_state =7, nthread = -1)
'''
model = xgb.XGBRegressor(colsample_bytree=0.70, gamma=0.005, 
                             learning_rate=0.02, max_depth=11, 
                             min_child_weight=1.1, n_estimators=3500,
                             reg_alpha=0.6, reg_lambda=1,
                             subsample=0.625, silent=1,
                             random_state =7, nthread = -1)
'''


param_dist = {
        ' reg_alpha':[0.3,0.4,0.5,0.6]
        }

model = xgb.XGBRegressor(n_estimators=3500,max_depth=9,
                         min_child_weight=1.222,reg_lambda=0.8,reg_alpha=0.6,
                         random_state =7,nthread = -1,gamma=0.023,subsample=0.5213,colsample_bytree=0.5)
grid = GridSearchCV(model,param_dist,cv = 3,scoring = 'r2',n_jobs=-1)
grid.fit(X_train, y_train)
best_estimator = grid.best_estimator_
print(best_estimator)
#輸出最優訓練器的精度
print(grid.best_score_)
print(grid.best_params_)
print('end:',datetime.datetime.now())
'''
model.fit(X_train, y_train)
#scores = cross_val_score(model, X_train, y_train, cv=5)
#print(scores)
y_pred = model.predict(X_test)
y_pred = np.exp(y_pred)
AVM(y_test, y_pred)