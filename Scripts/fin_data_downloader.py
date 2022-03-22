import time
import datetime
import pandas as pd

def downloadFinData(saveFilePath,ticker,year,month,date,hour,minute):

    period1_1 = datetime.datetime(year,month,date,hour,minute)
    period1 = int(time.mktime(period1_1.timetuple()))
    period2 = int(time.time())

    query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval=1d&events=history&includeAdjustedClose=true'
    print(query_string)
    df = pd.read_csv(query_string)
    df.drop('Adj Close',axis=1,inplace=True)

    df.to_csv(saveFilePath, sep=',')




