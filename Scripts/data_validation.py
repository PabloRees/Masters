import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#heavySpeeches = pd.read_csv('/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Complete_data/heavy_final_dataset( 101566 , 21 ) .csv')
#mainSpeeches = pd.read_csv('/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Complete_data/final_dataset(106774, 237).csv')

def sanityCheck(heavySpeeches,mainSpeeches):
    npHeavy = heavySpeeches['Date'].to_numpy()
    npMain = mainSpeeches['Date'].to_numpy()

    uniqueHeavy,countsHeavy = np.unique(npHeavy, return_counts=True)
    uniqueMain,countsMain = np.unique(npMain,return_counts=True)

    print(len(uniqueHeavy))

    print(len(uniqueMain))

    print(uniqueMain[1700])
    print(uniqueHeavy[1700])


    for i in uniqueMain:
        if not i in uniqueHeavy:
            print(i)

def makeSpeechTags(df):
    tagList = []
    df.drop_duplicates(inplace=True)
    for i in range(len(df)):

        type = df['Type'].iloc[i]
        name = df['Name'].iloc[i]
        title = df['Title'].iloc[i]
        date = df['Date'].iloc[i]

        tag = f'{type}_{name}_{title}_{date}'
        tagList.append(tag)

    df['SpeechTags'] = tagList

    return df

def duplicateCheck(df):

    tooShort = 0
    wordTagList = []
    for i in df['No.Stops.Transcript']:
        list = i.split("'",1)[-1][:-1].split(",")
        if len(list) <50:
            tooShort +=1
            wordTagList.append('tooShort')
        else:

            firstFifty = []
            for j in range(0,50):
                firstFifty.append(f'_{list[j]}')
            wordTag = ''.join(firstFifty)

            wordTagList.append(wordTag)


    df['firstFifty'] = wordTagList

    wordTagArray = np.array(wordTagList)

    tagCount, tagFreq = np.unique(wordTagArray,return_counts=True)

    print(len(wordTagList))
    print(len(tagCount))
    print(f'There were {tooShort} speeches shorter than 50 words')
    print(f'Num duplicates = {len(wordTagList) - len(tagCount)}')

    return df

def tagHeavySpeeches(df):
    df.drop_duplicates(inplace=True)
    df = duplicateCheck(df)
    df = makeSpeechTags(df)

    columns = ['firstFifty','SpeechTags']
    for i in range(200):
        columns.append(f'DV_{i}')

    taggedDocVecs = df[columns]

    taggedDocVecs.to_csv('/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Complete_data/taggedDocVecs.csv')

def joiner():
    taggedDvs = pd.read_csv(
        '/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Complete_data/taggedDocVecs.csv')
    taggedDvs.drop(labels='Unnamed: 0', axis=1, inplace=True)

    taggedMain = pd.read_csv(
        '/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Complete_data/final_dataset_tagged.csv')
    taggedMain.drop(labels=['Unnamed: 0', 'Unnamed: 0.1'], axis=1, inplace=True)

    countMainTags, freqMainTags = np.unique(taggedMain['SpeechTags'].to_numpy(), return_counts=True)
    countDvsTags, freqDvsTags = np.unique(taggedDvs['SpeechTags'].to_numpy(), return_counts=True)

    print(f'There are {len(taggedMain)} values and {len(countMainTags)} unique speech tags in the main df')
    print(f'There are {len(taggedDvs)} values and {len(countDvsTags)} unique speech tags in the Dvs df')

    countDvsFiftys, freqDvsFiftys = np.unique(taggedDvs['firstFifty'].to_numpy(), return_counts=True)

    print(f'There are {len(countDvsFiftys)} unique fiftys in the Dvs df')

    taggedDvs.drop_duplicates(subset='firstFifty',inplace=True)
    taggedDvs.drop_duplicates(subset='SpeechTags',inplace=True)
    taggedMain.drop_duplicates(subset='SpeechTags',inplace=True)

    fullDf = pd.merge(taggedMain, taggedDvs, on='SpeechTags', how='right')
    fullDfFiftys = fullDf.firstFifty.unique()

    print(f'There are {len(fullDfFiftys)} unique fiftys and {len(fullDf)} entries in the fullDf')

    print(f'There are {len(fullDf) - len(taggedMain)} more entries in the full_df than the mainDf')

    print(fullDf.shape)
    print(taggedDvs.shape)
    print(taggedMain.shape)

    return fullDf

finalDf = joiner()
finalDf.to_csv(f'/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Complete_data/final_tagged_dataset{finalDf.shape}.csv')

