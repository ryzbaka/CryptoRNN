import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import pandas_datareader.data as web
import datetime as dt

cryptos=['BTC-USD','LTC-USD','ETH-USD']

start=dt.datetime(2017,7,31)

stop=dt.datetime(int(dt.datetime.now().year),int(dt.datetime.now().month),int(dt.datetime.now().day))

data={}
for crypto in cryptos:
    oo=web.DataReader(crypto,'yahoo',start,stop)
    data[f'{crypto}_date']=oo.index.values
    data[f'{crypto}_close']=oo['Adj Close'].values
    data[f'{crypto}_volume']=oo['Volume'].values
    data[f'{crypto}_100pma']=oo['Adj Close'].rolling(window=100,min_periods=0).mean().values
'''
print(data.keys())
for key,value in data.items():
    print(key,':',value)
'''
oo=pd.DataFrame.from_dict(data=data)
oo.set_index(oo['ETH-USD_date'],drop=True,inplace=True)
oo.drop(labels=['LTC-USD_date','BTC-USD_date'],axis=1,inplace=True)
#oo.head(10)
oo=oo.rename_axis('Dates')
#main_df[f'{RATIO_TO_PREDICT}_future']=main_df[f'{RATIO_TO_PREDICT}_close'].shift(-FUTURE_PERIOD_PREDICT)
FUTURE_PERIOD_PREDICT=1
for crypto in cryptos:
    oo[f'{crypto}_future']=oo[f'{crypto}_close'].shift(-FUTURE_PERIOD_PREDICT)

def classify(current,future):
    if float(future)>float(current):#if cost increases in future
        return 1
    else:
        return 0
    
for crypto in cryptos:
    oo[f'{crypto}_target']=list(map(classify,oo[f'{crypto}_close'],oo[f'{crypto}_future']))

for column in oo.columns:
    if 'future' in column:
        oo.drop(labels=[column],axis=1,inplace=True)

import numpy as np
key=f'{dt.datetime.now().year}_{dt.datetime.now().month}_{dt.datetime.now().day}'
print(f'csv file generated as cryptdata{key}.csv')
oo.to_csv(f'cryptdata{key}.csv')