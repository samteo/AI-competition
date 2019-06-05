# -*- coding: utf-8 -*-
"""
Created on Tue May 14 09:23:04 2019
df.columns[df.isnull().sum()>0]
df.isnull().sum()
df.columns[df.dtypes == 'object']
'town1', 'building_material1', 'city1', 'building_type1',
       'building_use1'],
      dtype='object')
df["town1"].value_counts()
df.columns[df.isna().any()].tolist()
 xgb.plot_importance(model,max_num_features = 5)
@author: Big data
"""

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
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

df_train_noparking = pd.read_csv('noparking.csv')
df_train_noparking ["house_age"]=(df_train_noparking["txn_dt"].sub(df_train_noparking ["building_complete_dt"]))/365  
df_train_noparking ["house_age_square"] = df_train_noparking ["house_age"]**2
y = df_train_noparking.pop('total_price')
df_test_noparking = pd.read_csv("noparking_test.csv")

df=pd.concat((df_train_noparking,df_test_noparking),axis=0)

for i in df.columns:
    if "index" in i:
        df[i]=df[i].astype("category")
df["city"]=df["city"].astype("category")
df["town"]=df["town"].astype("category")
df["village"]=df["village"].astype("category")
df["building_type"]=df["building_type"].astype("category")
#df['village']=df['village'].astype("category")

#df['datediff'] = df['txn_dt'] - df['building_complete_dt']
#df["house_age"]=df["txn_dt"].sub(df["building_complete_dt"])        
df["total_area"]=df['land_area'].add(df["building_area"])         
        
df['lat']=df['lat'].sub(-42)        
        
median = df['village_income_median'].median()
df['village_income_median']=df['village_income_median'].fillna(df['village_income_median'].median())
#有值"XIII_index_5000"]"XI_index_5000"]
#"V_index_5000","V_index_10000",
#df.drop(["N_10000","I_index_5000","I_index_10000","II_index_5000","II_index_10000","VI_10","building_id","parking_way","parking_price"],axis=1)
df = df.drop(["N_10000","I_index_5000","I_index_10000","II_index_5000","II_index_10000","III_index_5000","III_index_10000","IV_index_10000",
              "V_index_5000","V_index_10000","VI_index_5000","VI_index_10000","VII_index_5000","VII_index_10000","VIII_index_5000","VIII_index_10000",
              "IX_index_5000","IX_index_10000","X_index_5000","X_index_10000","XI_index_10000","XII_index_5000","XII_index_10000","XIV_index_5000",
              "XIV_index_10000","village",
              "VI_10","building_id","parking_way","parking_price"],axis=1)
#total_price1 = df_train_noparking ['total_price']
#df_train_noparking['total_price1'] = total_price1
#df_train_noparking = df_train_noparking.drop(["total_price"],axis=1)

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
            
            
"""
for i in X.columns:
    if X[i].dtypes == "int64" or X[i].dtypes == "float64":
        skewness1 = X[i].skew()
        if skewness1 < -0.75 or skewness1 > 0.75:
            X[i] = boxcox1p(X[i], 0.15)
            skewness2 = X[i].skew()
            print(i,skewness1,skewness2)
"""     
   
df=pd.get_dummies(df)
X_train_noparking = df[:30255]
X_test_noparking = df[30255:]
X_train, X_test, y_train, y_test = train_test_split(X_train_noparking, y, test_size=0.33, random_state=42)
#X_train = X_train[y_train<30000000]
#y_train = y_train[y_train<30000000]
y_train = np.log(y_train)


        
#df_filter = df[df['village_income_median'].notnull()].copy()
#total_price_d100000 = df_filter['total_price'].div(100000).round(2).copy()
#df_filter['total_price_d100000'] = total_price_d100000
#df_aaa = pd.DataFrame([])
#df_aaa['A'] = df_filter['village_income_median']
#df_aaa['B'] = df_filter['total_price_d100000']
#df_aaa = df_aaa[df_aaa['B']<10000]





#ax = sns.scatterplot(x='A', y="B",data = df_aaa)

#ax = sns.scatterplot(x='village_income_median', y="total_price_d100000",data = df_filter)

'''
df['village_income_median'] = df['village_income_median'].fillna(df['village_income_median'].mode(),inplace=True)


df = df.drop(["N_10000","I_index_5000","I_index_10000","building_id","parking_way","parking_price"],axis=1)
df = df.drop(df.columns[0],axis=1)

df=pd.get_dummies(df)
total_price1 = df ['total_price']
df['total_price1'] = total_price1
df = df.drop(["total_price"],axis=1)
sns.boxplot(x=df['total_price1'])
sns.barplot(x=df["city1"],y=df['total_price1'])
sns.distplot(df["XIV_MIN"])
sns.distplot(df["bachelor_rate"])
sns.distplot(df["doc_Rate"])
df["bachelor_rate"]=np.log(df["bachelor_rate"])
df["XIV_MIN"]=np.log(df["XIV_MIN"])
print(df["XIV_MIN"].skew())
print(df["doc_Rate"].skew())

X = df.iloc[:,0:-1]
y = df.iloc[:,-1]

#pca = PCA(n_components=200)
#pca.fit(X)
#pca_samples = pca.transform(X)
#X = pd.DataFrame(pca_samples)
'''

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

model = xgb.XGBRegressor(colsample_bytree=0.70, gamma=0.005, 
                             learning_rate=0.02, max_depth=9, 
                             min_child_weight=1.1, n_estimators=3500,
                             reg_alpha=0.6, reg_lambda=1,
                             subsample=0.625, silent=1,
                             random_state =7, nthread = -1)
'''
model = GradientBoostingRegressor(n_estimators=4500, learning_rate=0.03,
                                   max_depth=5, max_features='sqrt',
                                   min_samples_leaf=6, min_samples_split=50, 
                                   loss='huber', random_state =5)
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
#scores = cross_val_score(model, X_train, y_train, cv=3)
#print(scores)
y_pred = model.predict(X_test)
y_pred = np.exp(y_pred)
AVM(y_test, y_pred)
y_predfinal = model.predict(X_test_noparking)
y_predfinal= np.exp(y_predfinal)
df_predfinal=pd.DataFrame()
df_predfinal["building_id"] =df_test_noparking["building_id"]
df_predfinal["total_price"]= y_predfinal

df_predfinal.to_csv("noparking_result.csv",index=False)
#y_pred = np.exp(y_pred)

print('end:',datetime.datetime.now())
    
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


le = OneHotEncoder(catesparse = False)
aa= le.fit_transform(df['town1'].values.reshape(-1,1))
df = pd.concat([df,aa])'''