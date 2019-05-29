# -*- coding: utf-8 -*-
"""
Created on Wed May 29 10:14:10 2019

@author: SAM
"""

import pandas as pd
import numpy as np
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
    
df1 = pd.read_csv('train.csv')
df2 = pd.read_csv('test.csv')
y = df1.pop('total_price')
df=pd.concat((df1,df2),axis=0)

df =df.drop(["parking_area"],axis =1)
df['parking_price']=df['parking_price'].fillna(0)
df['txn_floor']=df['txn_floor'].fillna(df['total_floor'])
df['village_income_median']=df['village_income_median'].fillna(df['village_income_median'].median())


for i in df.columns:
    if "index" in i:
        df[i]=df[i].astype("category")
df["building_use"]=df["building_use"].astype("category")
df["city"]=df["city"].astype("category")
df["town"]=df["town"].astype("category")
df["building_material"]=df["building_material"].astype("category")
df["house_age"]=(df["txn_dt"].sub(df["building_complete_dt"]))/365  
df["house_age_square"] = df["house_age"]**2
df["total_area"]=df['land_area'].add(df["building_area"])
df["parking_way"]=df["parking_way"].astype("category")
df["building_type"]=df["building_type"].astype("category")
df['lat']=df['lat'].sub(-42)    
df['XIV_1050']=df["XIV_10"]+df["XIV_50"]

df = df.drop(["N_10000","I_index_5000","I_index_10000","II_index_5000","II_index_10000","III_index_5000","III_index_10000","IV_index_10000",
              "V_index_5000","V_index_10000","VI_index_5000","VI_index_10000","VII_index_5000","VII_index_10000","VIII_index_5000","VIII_index_10000",
              "IX_index_5000","IX_index_10000","X_index_5000","X_index_10000","XI_index_10000","XII_index_5000","XII_index_10000","XIV_index_5000",
              "XIV_index_10000","village","XIV_10","XIV_50",
              "VI_10","building_id"],axis=1)
    
for i in df.columns:
    if df[i].dtypes == "int64" or df[i].dtypes == "float64":
        skewness1 = df[i].skew()
        if  skewness1 >1:
            df[i] = np.log1p(df[i])
            skewness2 = df[i].skew()
            print(i,skewness1,skewness2)
        elif skewness1 < -1:
            df[i] = np.cbrt(df[i])
            skewness2 = df[i].skew()
            print(i,skewness1,skewness2)
            
df=pd.get_dummies(df)
X_train_noparking = df[:60000]
X_test_noparking = df[60000:]
X_train, X_test, y_train, y_test = train_test_split(X_train_noparking, y, test_size=0.33, random_state=42)
#X_train = X_train[y_train>500000]
#y_train = y_train[y_train>500000]
y_train = np.log(y_train)            


print('start:',datetime.datetime.now())
model = xgb.XGBRegressor(colsample_bytree=0.70, gamma=0.005, 
                             learning_rate=0.02, max_depth=20, 
                             min_child_weight=1.1, n_estimators=4000,
                             reg_alpha=0.6, reg_lambda=0.5,
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
y_predfinal = model.predict(X_test_noparking)
y_predfinal= np.exp(y_predfinal)
df_predfinal=pd.DataFrame()
df_predfinal["building_id"] =df2["building_id"]
df_predfinal["total_price"]= y_predfinal

df_predfinal.to_csv("noparking_result.csv",index=False)
#y_pred = np.exp(y_pred)

print('end:',datetime.datetime.now())