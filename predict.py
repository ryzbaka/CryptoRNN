from sklearn import preprocessing
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import datetime as dt
import pandas_datareader.data as web
from warnings import simplefilter

simplefilter(action='ignore')

banner='''
	  /$$$$$$                                  /$$                        
	 /$$__  $$                                | $$                        
	| $$  \__/  /$$$$$$  /$$   /$$  /$$$$$$  /$$$$$$    /$$$$$$  /$$$$$$$ 
	| $$       /$$__  $$| $$  | $$ /$$__  $$|_  $$_/   /$$__  $$| $$__  $$
	| $$      | $$  \__/| $$  | $$| $$  \ $$  | $$    | $$  \ $$| $$  \ $$
	| $$    $$| $$      | $$  | $$| $$  | $$  | $$ /$$| $$  | $$| $$  | $$
	|  $$$$$$/| $$      |  $$$$$$$| $$$$$$$/  |  $$$$/|  $$$$$$/| $$  | $$
 	\______/ |__/       \____  $$| $$____/    \___/   \______/ |__/  |__/
	                     /$$  | $$| $$                                    
 	                   |  $$$$$$/| $$                                    
 	                    \______/ |__/            
       '''
print(banner)
print('Select a cryptocurrency ticker:')
print('1. BTC-USD - Bitcoin-US Dollar')
print('2. LTC-USD - Litecoin-US Dollar')
print('3. ETH-USD - Ethereum-US Dollar')
uin=int(input('>'))

if uin==1:
	ticker='BTC-USD'
elif uin==2:
	ticker='LTC-USD'
elif uin==3:
	ticker='ETH-USD'
else:
	print('invalid option')
	print('setting ticker to BTC-USD')
	ticker='BTC-USD'


data={}
start=dt.datetime(2019,6,int(dt.datetime.now().day)-10)
stop=dt.datetime(2019,6,int(dt.datetime.now().day))
cryptos=['BTC-USD','LTC-USD','ETH-USD']
for crypto in cryptos:
    oo=web.DataReader(crypto,'yahoo',start,stop)
    data[f'{crypto}_date']=oo.index.values
    data[f'{crypto}_close']=oo['Adj Close'].values
    data[f'{crypto}_volume']=oo['Volume'].values
    #data[f'{crypto}_100pma']=oo['Adj Close'].rolling(window=100,min_periods=0).mean().values

lens=[]
for i in data.keys():
    lens.append(len(data[i]))

minlen=min(lens)
for i in data.keys():
    data[i]=data[i][:minlen]
oo=pd.DataFrame.from_dict(data=data)
for i in oo.columns.values:
    if 'date' in i or '100pma' in i:
        oo.drop(labels=i,axis=1,inplace=True)

oo=oo.iloc[-10:,:]
for col in oo.columns:
    oo[col]=oo[col].pct_change()# we're trying to understand the percentage change of each crypto's price
    oo.fillna(value=0,inplace=True)
    oo[col]=preprocessing.scale(oo[col].values)
x=oo.values
#x.shape
x=x.reshape(1,x.shape[0],x.shape[1])
print('shape',x.shape)
model_filepath='auto_models/auto_RNN_Final-10-0.800.model'
model=load_model(model_filepath)
prediction=model.predict(x)

if prediction[0][1]>prediction[0][0]:
	print('############')
	print('#Do not buy#')
	print('############')
else:
	print('#####')
	print('#Buy#')
	print('#####')


print(prediction,prediction.shape)
