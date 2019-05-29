# -*- coding: utf-8 -*-
"""
Created on Tue May 21 18:47:20 2019

@author: Big data
"""
import pandas as pd
df_withparking_result = pd.read_csv('withparking_result.csv')
df_nohparking_result = pd.read_csv('noparking_result.csv')
df_nohparking_result1 = pd.read_csv('noparking1_result.csv')
result = pd.concat((df_withparking_result ,df_nohparking_result,df_nohparking_result1),axis = 0)
result.to_csv('submit_test.csv',index=False)