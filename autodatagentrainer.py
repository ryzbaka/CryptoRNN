#############################################################################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~DATA PROCURING~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#############################################################################################

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
'''
for column in oo.columns:
    if 'future' in column:
        oo.drop(labels=[column],axis=1,inplace=True)
'''
import numpy as np
key=f'{dt.datetime.now().year}_{dt.datetime.now().month}_{dt.datetime.now().day}_{np.random.rand(1)[0]}'
#print(type(list(oo.index.values)[0]),list(oo.index.values)[0])
print(f'csv file generated as cryptdata{key}.csv')
oo.to_csv(f'cryptdata{key}.csv')
datafile=f'cryptdata{key}.csv'
#############################################################################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~DATA PREPROCESSING~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#############################################################################################
import time
from collections import deque# its stack that maintains its size equal to the maxLen
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import preprocessing
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,LSTM,CuDNNLSTM,BatchNormalization
from tensorflow.keras.callbacks import TensorBoard,ModelCheckpoint

style.use('ggplot')

SEQ_LENGTH=10 # we're using the last 10 days of data to make a prediction
FUTURE_PERIOD_PREDICT=1 # every period in this data is 1 day so we'll predict for the next 1 day.
RATIO_TO_PREDICT="BTC-USD"
EPOCHS=10
BATCH_SIZE=64
NAME=f'{SEQ_LENGTH}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}'


def preprocess_df(df,RATIO_TO_PREDICT):
        df=df.drop(f'{RATIO_TO_PREDICT}_future',axis=1)# don't leave in any feature columns during the training of the model
        for col in df.columns:
                if col!=f'{RATIO_TO_PREDICT}_target': #normalize all except for the target itself.
                        df[col]=df[col].pct_change()# we're trying to understand the percentage change of each crypto's price
                        df.dropna(inplace=True)
                        df[col]=preprocessing.scale(df[col].values)
        df.dropna(inplace=True)#juuustt in case lol
        #print(df.head())
        sequential_data=[]
        prev_days=deque(maxlen=SEQ_LENGTH)#wait till this prev_days actually has 60 values
        
        for i in df.values:
                prev_days.append([n for n in i[:-1]])# taking all values except target, make sure you fix this for a dataset that contains target for all cryptos in GetData.py
                if len(prev_days)==SEQ_LENGTH:
                        sequential_data.append([np.array(prev_days),i[-1]]) #tagging a target label for each value

        random.shuffle(sequential_data)
        #now we have our sequences and targets so now we'll have to balance our data so that don't keep predicting the majority class
        buys=[]
        sells=[]

        for seq,target in sequential_data:
                if target==0:
                        sells.append([seq,target])
                elif target==1:
                        buys.append([seq,target])
        random.shuffle(buys)
        random.shuffle(sells)
        lower=min(len(buys),len(sells))

        buys=buys[:lower]
        sells=sells[:lower]

        sequential_data=buys+sells

        random.shuffle(sequential_data)

        X=[]
        y=[]
        for seq,target in sequential_data:
                X.append(seq)
                y.append(target)
        return np.array(X),y


def classify(current,future):
    if float(future)>float(current):#if cost increases in future
        return 1
    else:
        return 0


pd.set_option('display.max_colwidth',-1)

base_df=pd.read_csv(datafile)

main_df=base_df.drop(labels=['BTC-USD_100pma','LTC-USD_100pma','ETH-USD_date','ETH-USD_100pma'],axis=1)
for i in main_df.columns.values:
    if 'target' in i:
        if f'{RATIO_TO_PREDICT}' not in i:
            main_df.drop(labels=i,axis=1,inplace=True)
for i in main_df.columns.values:
    if 'future' in i:
        if f'{RATIO_TO_PREDICT}' not in i:
            main_df.drop(labels=i,axis=1,inplace=True)

main_df.set_index('Dates',inplace=True)

times=sorted(main_df.index.values)#we are taking the last 5% of data
last_5pct=sorted(main_df.index.values)[-int(0.05*len(times))]# threshold of the last 5% of times

#main_df.set_index(main_df.Date,inplace=True)
validation_main_df=main_df[(main_df.index>=last_5pct)]
main_df=main_df[(main_df.index.values<last_5pct)]

train_x,train_y=preprocess_df(main_df,RATIO_TO_PREDICT)
validation_x,validation_y=preprocess_df(validation_main_df,RATIO_TO_PREDICT)

print(f'train data:{len(train_x)} validation: {len(validation_x)}')
print(f"Don't buys: {train_y.count(0)}, buys: {train_y.count(1)}")
print(f'VALIDATION dont buys: {validation_y.count(0)}, buys:{validation_y.count(1)}')

#############################################################################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~MODEL TRAINING~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#############################################################################################

model=Sequential()
model.add(LSTM(128,input_shape=(train_x.shape[1:]),return_sequences=True,activation='tanh'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128,input_shape=(train_x.shape[1:]),return_sequences=True,activation='tanh'))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(LSTM(128,input_shape=(train_x.shape[1:]),activation='tanh'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(2,activation='softmax'))

optimizer=tf.keras.optimizers.Adam(lr=0.001,decay=1e-6)

model.compile(loss='sparse_categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])

tensorboard=TensorBoard(log_dir=f'logs/{NAME}')

filepath='RNN_Final-{epoch:02d}-{val_acc:.3f}'#unique file name that will include the epoch and the validation acc for that epoch
checkpoint=ModelCheckpoint("models/{}.model".format(filepath,monitor='val_acc',verbose=1,save_best_only=True,mode='max'))#saves the best ones


history=model.fit(train_x,train_y,batch_size=BATCH_SIZE,epochs=EPOCHS,validation_data=(validation_x,validation_y),callbacks=[tensorboard,checkpoint])
