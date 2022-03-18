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

#downloadFinData('/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/GSPC_1.csv','^GSPC',1927, 12, 30, 23, 59)
#downloadFinData('/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/MetaData/USDX.csv','DX-Y.NYB',1971, 1, 3, 23, 59)
#downloadFinData('/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/MetaData/BTC.csv','BTC-USD',2014, 9, 16, 23, 59)
#downloadFinData('/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/MetaData/VIX.csv','^VIX',1990, 1, 1, 23, 59)
#downloadFinData('/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/MetaData/OIL.csv','CL%3DF',2000, 8, 22, 23, 59)
#downloadFinData('/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/MetaData/NASDAQ_comp.csv','%5EIXIC',1971, 2, 4, 23, 59)
downloadFinData('/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/MetaData/SSE_comp.csv','000001.SS',1997, 7, 1, 23, 59)


