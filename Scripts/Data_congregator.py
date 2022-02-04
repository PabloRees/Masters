import pandas as pd
import os

def combineSpeeches(speech_load_file_path):
    speechTypeList = os.listdir(speech_load_file_path)
    df = pd.read_csv(speech_load_file_path + '/' + speechTypeList[0], sep='\t')
    speechTypeList.pop(0)

    for i in speechTypeList:
        if not i.startswith('.'):
            print(i)
            df_to_add = pd.read_csv(speech_load_file_path + '/' +i , sep='\t')
            df = pd.concat([df,df_to_add], axis=0,ignore_index=True)

    df['Date'] = pd.to_datetime(df.Date)
    df.sort_values(by=['Date'],ignore_index=True,inplace=True)
    df.reindex
    #df.to_csv(speech_load_file_path+'/'+'combineSortTest.tsv',sep = '\t')

    return df

def combine_GSPC_Speeches(speechDf, GSPCDf):
    for i in range(len(speechDf['Date'])):
        speechDf['Date'][i] = pd.to_datetime(speechDf['Date'][i].split('+', 1)[0])
        if speechDf['Date'][i].hour > 16:
            speechDf['Date'][i] = speechDf['Date'][i].replace(day=df_speech_test['Date'][i].day + 1)

    df_speech_test['Date'] = pd.to_datetime(df_speech_test['Date']).dt.date
    df_speech_test['Date'] = pd.to_datetime(df_speech_test['Date'])

    GSPC_df['Date'] = pd.to_datetime(GSPC_df.Date)

    full_df = pd.merge_asof(left = speechDf.sort_values('Date'),right = GSPCDf, on='Date', direction="forward")
    full_df.to_csv('/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Speech_data_test/combinedGSPC_speech.tsv',sep = '\t')

    return full_df

df_speech_test = pd.read_csv("/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Speech_data/convention-speeches(60, 6).tsv", sep= '\t')
GSPC_df = pd.read_csv('/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/GSPC.csv',sep = ',')
speech_load_file_path = '/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Speech_data'

#combine_GSPC_Speeches(df_speech_test,GSPC_df)

#combineSpeeches(speech_load_file_path)
