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

SEQ_LENGTH=60 # we're using the last 60 minutes of data to make a prediction
FUTURE_PERIOD_PREDICT=3 # every period in this data is 1 minute so we'll predict for the next 3 minutes.
RATIO_TO_PREDICT="BCH-USD"
EPOCHS=20
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

ratios=['BTC-USD','LTC-USD','BCH-USD','ETH-USD']
main_df=pd.DataFrame()
for ratio in ratios:
    close=f'{ratio}_close'
    volume=f'{ratio}_volume'
    dataset=f'crypto_data/crypto_data/{ratio}.csv'
    df=pd.read_csv(dataset,names=['low','high','open','close','volume'])
    df.rename(columns={'close':close,'volume':volume},inplace=True)
    df=df[[close,volume]]
    if len(main_df)==0:
        main_df=df
    else:
        main_df=main_df.join(df)

main_df[f'{RATIO_TO_PREDICT}_future']=main_df[f'{RATIO_TO_PREDICT}_close'].shift(-FUTURE_PERIOD_PREDICT)
# the shift line moves the close column of the selected ratio by 3 places.
# that way the future column shows the closing price for the ratio that showed up after 
# 3 minutes.
#print(main_df[[f'{RATIO_TO_PREDICT}_close',f'{RATIO_TO_PREDICT}_future']].head())
#print('...')
#print(main_df[[f'{RATIO_TO_PREDICT}_close',f'{RATIO_TO_PREDICT}_future']].tail())

main_df[f'{RATIO_TO_PREDICT}_target']=list(map(classify,main_df[f'{RATIO_TO_PREDICT}_close'],main_df[f'{RATIO_TO_PREDICT}_future']))
#print(main_df.head(10))

times=sorted(main_df.index.values)#we are taking the last 5% of data
last_5pct=sorted(main_df.index.values)[-int(0.05*len(times))]# threshold of the last 5% of times

validation_main_df=main_df[(main_df.index>=last_5pct)]
main_df=main_df[(main_df.index.values<last_5pct)]
train_x,train_y=preprocess_df(main_df,RATIO_TO_PREDICT)
validation_x,validation_y=preprocess_df(validation_main_df,RATIO_TO_PREDICT)


print(f'train data:{len(train_x)} validation: {len(validation_x)}')
print(f"Don't buys: {train_y.count(0)}, buys: {train_y.count(1)}")
print(f'VALIDATION dont buys: {validation_y.count(0)}, buys:{validation_y.count(1)}')


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
print('*'*10)
print(f'{train_x.shape[1:]} is the size of the input tensor')
print('*'*10)

filepath='RNN_Final-{epoch:02d}-{val_acc:.3f}'#unique file name that will include the epoch and the validation acc for that epoch
checkpoint=ModelCheckpoint("models/{}.model".format(filepath,monitor='val_acc',verbose=1,save_best_only=True,mode='max'))#saves the best ones


history=model.fit(train_x,train_y,batch_size=BATCH_SIZE,epochs=EPOCHS,validation_data=(validation_x,validation_y),callbacks=[tensorboard,checkpoint])
