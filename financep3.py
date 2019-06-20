import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as dt
import datetime as dt
from matplotlib import style 

style.use('ggplot')

df=pd.read_csv('BTC_USD.csv',parse_dates=True,index_col=0)

df['100pma']=df['Adj Close'].rolling(window=100,min_periods=0).mean() #100 point moving average calculation
# the min_periods decides how many previous values you need to calculate the rolling value



print(df.head())

ax1=plt.subplot2grid((6,1),(0,0),rowspan=5,colspan=1)
ax2=plt.subplot2grid((6,1),(5,0),rowspan=1,colspan=2,sharex=ax1)

ax1.plot(df.index, df['Adj Close'])
ax1.plot(df.index, df['100pma'])
ax2.bar(df.index, df['Volume'])

plt.show()
