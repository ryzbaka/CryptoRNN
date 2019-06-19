import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web

style.use('ggplot')


start=dt.datetime(2000,1,1)
#YYYY-MM-DD
stop=dt.datetime(2018,12,31)

#the ticker for the company
#Tesla : TSLA
#The 'yahoo' parameter gets data from yahoo finance
df=web.DataReader('BTC-USD','yahoo',start,stop)
print(df.head())
plt.plot(df.index,df.Close,color='green')
plt.title('BTC-USD')
plt.show()