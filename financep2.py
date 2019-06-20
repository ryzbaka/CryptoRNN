import pandas_datareader.data as web
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

start=dt.datetime(2009,1,1)
stop=dt.datetime(2019,1,1)

ticker='BTC-USD'

df=web.DataReader(ticker,'yahoo',start,stop)

df.to_csv('BTC_USD.csv')

df.plot()
plt.show()