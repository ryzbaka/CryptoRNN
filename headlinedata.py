import re
import datetime as dt
import pandas as pd
from newsapi import NewsApiClient
import numpy as np

api=NewsApiClient(api_key='eab4212230654aa7810d23ade738c466')

search_words=['bitcoin','litecoin','ethereum','cryptocurrency','crypto']

data=[]
for word in search_words:
    try:
        data.append(api.get_everything(q=word))
    except Exception as e:
        print(str(e))
def process_data():
    dict_data={}
    headline=[]
    date=[]
    for keyword in data:
        if keyword['status']=='ok':
            for article in keyword['articles']:
                headline.append(article['title'])
                date.append(article['publishedAt'])
    dict_data['Headlines']=headline
    dict_data['Date']=date
    df=pd.DataFrame.from_dict(data=dict_data)

    return df
df=process_data()
def process_date(x):
    datestring=str(x)
    vals=datestring[:10].split('-')
    return np.datetime64(dt.datetime(int(vals[0]),int(vals[1]),int(vals[2])))
df.Date=df.Date.map(lambda x: process_date(x))
df.set_index(df.Date,inplace=True)
df.drop(labels='Date',inplace=True,axis=1)
df.Headlines=df.Headlines.apply(lambda x: x.lower())
df.Headlines=df.Headlines.apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))
key=f'{dt.datetime.now().year}_{dt.datetime.now().month}_{dt.datetime.now().day}'
#print(type(list(df.index.values)[0]),list(oo.index.values)[0])
try:
    print(f'csv file generated as headlines{key}.csv')
    df.to_csv(f'headlines{key}.csv')
except:
    print('file already_exists')
    print(f'csv file generated as headlines_updated{key}.csv')
    df.to_csv(f'headlines_updated{key}.csv')
