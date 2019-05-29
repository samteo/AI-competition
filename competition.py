# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np

df = pd.read_csv('train.csv')
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
df_noparking.to_csv("noparking.csv",index=False)
df_noparking_f = df[(df['parking_way']==2) & (df['txn_floor'].isnull())]
df_noparking_f.to_csv("noparking1.csv",index=False)
df_withparking = df[df['parking_way']!=2]
df_withparking.to_csv("withparking.csv",index=False)



'''
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, skew

df = pd.read_csv('train.csv')

#df['town'].unique()
#df['town'].nunique()
#df.groupby(['town'])['building_id'].nunique()


def trans_num2str(column):
    s = ''
    dict_code = {
        "0": "A",
        "1": "B",
        "2": "C",
        "3": "D",
        "4": "E",
        "5": "F",
        "6": "G",
        "7": "H",
        "8": "I",
        "9": "J"
    }
    col_name = str(column) + str(1)
    new_col = df[column].astype(str).copy()
    for i in range(len(new_col)):
        new_col[i] = list(new_col[i])
        new_col[i] = [dict_code[k] for k in new_col[i]]
        new_col[i] = s.join(new_col[i])
    df[col_name] = new_col


trans_num2str("town")
trans_num2str("building_material")
trans_num2str("city")
trans_num2str("building_type")
trans_num2str("building_use")
df = df.drop(
    ["town", "building_material", "city", "building_type", "building_use"],
    axis=1)
df = df.drop(["parking_area"], axis=1)  #缺太多


df_noparking = df[df['parking_way'] == 2]
'''
f, ax = plt.subplots(figsize =(10,10))
ax.set_xlim([0, 5])
#ax.set(xlim=(100,1000))
sns.distplot(df_noparking['total_price'],fit=norm)
plt.ticklabel_format(style='sci',scilimits=(0,0))
'''


total_price = df_noparking['total_price']/1000000
f, ax = plt.subplots(figsize =(10,10))
sns.distplot(total_price,fit=norm,ax=ax)
ax.set(xlim=(0,10))
(mu, sigma) = norm.fit(df_noparking['total_price'])
print(mu,sigma)
print(df_noparking['total_price'].skew())
total_price_log = np.log1p(df_noparking['total_price'])
print(total_price_log.skew())
pd.options.display.float_format = '{:,.2f}'.format


sns.distplot(df_noparking['XIV_5000'] )
'''