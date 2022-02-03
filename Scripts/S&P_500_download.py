import time
import datetime
import pandas as pd

period1 = int(time.mktime(datetime.datetime(1927, 12, 30, 23, 59).timetuple()))
period2 = int(time.time())

query_string = f'https://query1.finance.yahoo.com/v7/finance/download/^GSPC?period1={period1}&period2={period2}&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true'

df = pd.read_csv(query_string)
df.drop('Adj Close',axis=1,inplace=True)

print(df)

df.to_csv("/Users/pablo/Desktop/Masters /Github Repository/Masters/Data/GSPC.csv", sep=',')