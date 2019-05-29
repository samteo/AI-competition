# -*- coding: utf-8 -*-
"""
Created on Tue May 21 14:30:55 2019

@author: Big data
"""

import pandas as pd
import numpy as np

df = pd.read_csv('test.csv')
#df['town'].unique()
#df['town'].nunique()
#df.groupby(['town'])['building_id'].nunique()
'''
def trans_num2str(column):
    s=''
    dict_code = {"0":"A","1":"B","2":"C","3":"D","4":"E","5":"F","6":"G","7":"H","8":"I","9":"J"}
    col_name = str(column) + str(1)
    new_col = df[column].astype(str).copy()
    for i in range(len(new_col)):
        new_col[i] = list(new_col[i])
        new_col[i] = [dict_code[k] for k in new_col[i]]
        new_col[i] = s.join(new_col[i])
    df[col_name] = new_col
    df[col_name] = df[col_name].astype('category')

trans_num2str("town")
trans_num2str("building_material")
trans_num2str("city")
trans_num2str("building_type")
trans_num2str("building_use")
df=df.drop(["town","building_material","city","building_type","building_use"],axis =1)
'''
df =df.drop(["parking_area"],axis =1) #缺太多


df_noparking = df[(df['parking_way']==2) & (df['txn_floor'].notnull())]
df_noparking.to_csv("noparking_test.csv",index=False)
df_noparking1 = df[(df['parking_way']==2) & (df['txn_floor'].isnull())]
df_noparking1.to_csv("noparking1_test.csv",index=False)
df_withparking = df[df['parking_way']!=2]
df_withparking.to_csv("withparking_test.csv",index=False)