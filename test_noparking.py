# -*- coding: utf-8 -*-
"""
Created on Tue May 21 14:34:41 2019

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

df_test_noparking = pd.read_csv('noparking_test.csv')
for i in df_test_noparking.columns:
    if "index" in i:
        df_test_noparking[i]=df_test_noparking[i].astype("category")
        
   

df_test_noparking['village_income_median'] = df_test_noparking['village_income_median'].fillna(df_test_noparking['village_income_median'].mode())

df_buildingid_noparking = df_test_noparking.pop("building_id")
df_test_noparking = df_test_noparking.drop(["N_10000","I_index_5000","I_index_10000","parking_way","parking_price"],axis=1)
df_test_noparking = df_test_noparking.drop(df_test_noparking.columns[0],axis=1)

for i in df_test_noparking.columns:
    if df_test_noparking[i].dtypes == "int64" or df_test_noparking[i].dtypes == "float64":
        skewness1 = df_test_noparking[i].skew()
        if  skewness1 > 0.75:
            df_test_noparking[i] = np.log1p(df_test_noparking[i])
            skewness2 = df_test_noparking[i].skew()
            print(i,skewness1,skewness2)
        elif skewness1 < -0.75:
            df_test_noparking[i] = np.cbrt(df_test_noparking[i])
            skewness2 = df_test_noparking[i].skew()
            print(i,skewness1,skewness2)
df_test_noparking=pd.get_dummies(df_test_noparking)
df_test_noparking.to_csv("finaltest_noparking.csv",index=False)
df_buildingid_noparking.to_csv("buildingid_noparking.csv",index=False)