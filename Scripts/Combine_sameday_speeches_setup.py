import pandas as pd
import numpy as np
from data_validation import tagHeavySpeeches
from Doc2Vec import trainDoc2Vec, create_Doc_vectors, getVecs
from gensim.models.doc2vec import Doc2Vec
from datetime import datetime, timedelta

def initialSetup():
    df = pd.read_csv(
        '/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Complete_data/combinedHeavySpeeches.tsv', sep='\t')
    df.rename(columns={'No Stops Transcript': 'No_Stops_Transcript'}, inplace=True)
    df = tagHeavySpeeches(df)
    return df

def createGetDVs(df,filePath,combine_sameday_speeches):

    combined_PVDM_DV_20_Model = trainDoc2Vec(filePath,20,combine_sameday_speeches=combine_sameday_speeches
                                             ,model_type='PV_DM')

    combined_PVDBOW_DV_20_Model = trainDoc2Vec(filePath,20,combine_sameday_speeches=combine_sameday_speeches
                                               ,model_type='PV_DBOW')

    #combined_PVDBOW_DV_20_Model = Doc2Vec.load('Doc2Vec_20_combined_PV_DBOW.model')
    #combined_PVDM_DV_20_Model = Doc2Vec.load('Doc2Vec_20_combined_PV_DM.model')

    uniqueDatesDf = df.drop_duplicates(subset='Date')

    print(uniqueDatesDf.describe())

    combined_PVDM_DV_20 = create_Doc_vectors(uniqueDatesDf,combined_PVDM_DV_20_Model,combine_sameday_speeches=combine_sameday_speeches)
    combined_PVDBOW_DV_20 = create_Doc_vectors(uniqueDatesDf,combined_PVDBOW_DV_20_Model,combine_sameday_speeches=combine_sameday_speeches)

    uniqueDatesDf['combined_PVDM_DV_20'] = combined_PVDM_DV_20
    uniqueDatesDf['combined_PVDBOW_DV_20'] = combined_PVDBOW_DV_20

    uniqueDatesDf.to_csv(filePath)

    return uniqueDatesDf

def convertDay(Date):

    Date, Time = Date.split(' ', 1)
    Hour = int(Time.split(':', 1)[0])
    Date = pd.to_datetime(Date, format='%Y-%m-%d')
    if Hour > 15:
        Date += timedelta(days=1)

    if Date.weekday() == 5:
        Date += timedelta(days=2)
    elif Date.weekday() == 6: Date += timedelta(days=1)

    Date = str(Date).split(' ',1)[0]

    return Date

def Merge(df):
    GSPC = pd.read_csv('/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Complete_data/GSPC_features.csv')
    Meta = pd.read_csv('/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Complete_data/metadata.csv')

    df = GSPC.merge(df, 'outer', on='Date',indicator=True)
    df = df.merge(Meta, 'outer', on='Date')

    return df

def interpolateDVs(df):

    daysSinceLastSpeech = []
    date = pd.to_datetime(df['Date'].iloc[0], format='%Y-%m-%d')
    DV_20_PVDM = df.iloc[0]['combined_PVDM_DV_20']
    DV_20_PVDBOW = df.iloc[0]['combined_PVDBOW_DV_20']
    print(type(DV_20_PVDBOW))
    for i in range(len(df)):

        if type(df['combined_PVDM_DV_20'].iloc[i]).__module__ == np.__name__:
            daysSinceLastSpeech.append(0)
            date = pd.to_datetime(df['Date'].iloc[i],format='%Y-%m-%d')
            DV_20_PVDM = df['combined_PVDM_DV_20'].iloc[i]
            DV_20_PVDBOW = df['combined_PVDBOW_DV_20'].iloc[i]

        else:
            print(type(df['combined_PVDM_DV_20'].iloc[i]))
            daysSinceLastSpeech.append((pd.to_datetime(df.iloc[i]['Date'],format='%Y-%m-%d') - date).days)
            df.at[i, 'combined_PVDM_DV_20'] = DV_20_PVDM
            df.at[i, 'combined_PVDBOW_DV_20'] = DV_20_PVDBOW


    df['Days_since_last_speech'] = daysSinceLastSpeech

    return df


def transpose(df, column, label, vecLen):
    colNames = []
    for j in range(0, vecLen):
        colNames.append(f'{label}_{j}')

    nanRow = []
    for i in range(0,20):
        nanRow.append(np.nan)

    vecList = [nanRow]

    for i in range(1, len(df[column])):
        vec = df[column].iloc[i]
        vec = vec.replace('[', '').replace(']', '')
        vec = vec.lstrip().rstrip()
        vec = vec.replace('   ', ' ').replace('  ', ' ').replace('\n', '').replace('    ', ' ').replace('   ',
                                                                                                        ' ').replace(
            '  ', ' ')
        vec = vec.split(' ')

        floatVec = []
        for k in vec: floatVec.append(float(k))

        if not len(floatVec) == vecLen:
            print(i)
            print(df[column].iloc[i])
            print(floatVec)
        else:
            vecList.append(floatVec)

    vec_df = pd.DataFrame(data=vecList,columns=colNames)

    print(vec_df.head())

    df.drop(column, inplace=True, axis=1)

    df_all_cols = pd.concat([df, vec_df], axis=1)

    return df_all_cols


finalFilePath = '/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Combine_sameday_speeches/Combine_sameday_speeches_final.csv'

df = pd.read_csv(finalFilePath)

#print(type(df['combined_PVDM_DV_20'].iloc[1]))
#print(df['combined_PVDBOW_DV_20'].iloc[-1])

df = transpose(df,'combined_PVDM_DV_20','PVDM',20)
df = transpose(df,'combined_PVDBOW_DV_20','PVDBOW',20)

df.to_csv('/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Combine_sameday_speeches/Combine_sameday_speeches_final_transposed.csv')

exit()
filePath = '/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Combine_sameday_speeches/Combine_sameday_speeches2.csv'

df = pd.read_csv('/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Tagged_Raw_Speeches(84155, 8).csv')

for i in df.columns:
    if 'Unnamed' in i:
        df.drop(labels=[i],axis=1,inplace=True)
    if 'No' in i:
        if 'Stops' in i:
            if 'Transcript' in i:
                df.rename(columns={str(i):'No_Stops_Transcript'},inplace=True)

#df.to_csv(filePath)
df['Date'] = df['Date'].apply(convertDay)

df.to_csv(filePath)

df = createGetDVs(df,filePath,combine_sameday_speeches=True)

df = Merge(df)

df = interpolateDVs(df)

df.to_csv('/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Combine_sameday_speeches/Combine_sameday_speeches_final.csv')

print(df.describe())


