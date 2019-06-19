import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

SEQ_LENGTH=60 # we're using the last 60 minutes of data to make a prediction
FUTURE_PERIOD_PREDICT=3 # every period in this data is 1 minute so we'll predict for the next 3 minutes.

RATIO_TO_PREDICT="BTC-USD"

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

times=sorted(main_df.index.values)
#taking the last 5% of time data to test our model.
last_5pct=int(0.05*len(times))
last_5pct=times[-last_5pct:])
times=times[:-last_5pct]


