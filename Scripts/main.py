import os

import pandas as pd

from WebscrapeAndClean import *
from data_validation import *
import gensim.models
from SentimentAnalysis import *
from Word2Vec import *
from Doc2Vec import *
from fin_data_downloader import *
from R_Interface import *
from Data_congregator import *
from Transpose import *

############# Setting up text data #############
def setUpTextData():
    #speechDataSavePath = '/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Speech_data'

    #runWSC(speechDataSavePath)  # calls the WebScrapeAndClean process - text data

    #combinedSpeeches = combineSpeeches(speechDataSavePath)
    #print(f'Speeches combined')

    #if not 'No_Stops_Transcript' in combinedSpeeches.columns:
    #    for i in combinedSpeeches.columns:
    #        if 'No' in str(i):
    #            if 'Stops' in str(i):
    #               if 'Transcript' in str(i):
    #                   combinedSpeeches.rename(columns={str(i): 'No_Stops_Transcript'}, inplace=True)

    #taggedSpeeches = tagHeavySpeeches(combinedSpeeches)

    #print(f'Speeches tagged and duplicates dropped')
    #taggededSpeechesSavePath = f'/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Tagged_Raw_Speeches/Tagged_Raw_Speeches{taggedSpeeches.shape}.csv'
    #taggedSpeeches.to_csv(taggededSpeechesSavePath)

    #Word2Vec_Model = trainWord2Vec('/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Tagged_Raw_Speeches')  # creates the word2vec model - text data
    #Doc2Vec_Model_200 = trainDoc2Vec('/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Tagged_Raw_Speeches',200)
    #Doc2Vec_Model_20 = trainDoc2Vec('/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Tagged_Raw_Speeches',20)

    #print(f'Gensim models trained')

    speechDataSavePath = '/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Tagged_Raw_Speeches'

    featuresFilePath = "/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Speech_data_with_features/"
    liteFeaturesFilesPath = '/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Speech_data_lite/'
    heavyFeaturesFilesPath = '/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Speech_data'

    textPrep(speechDataSavePath, featuresFilePath, liteFeaturesFilesPath)
    exit()
    completeDataFilePath = '/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Complete_data'

    combinedLiteSpeeches = combineSpeeches(liteFeaturesFilesPath)
    combinedLiteSpeeches.to_csv(completeDataFilePath + '/combinedLiteSpeeches.tsv', sep='\t')

    combinedHeavySpeeches = combineSpeeches(heavyFeaturesFilesPath)
    combinedHeavySpeeches.to_csv(completeDataFilePath + '/combinedHeavySpeeches.tsv', sep='\t')

############# Setting up NLP data #############
def textPrep(speechDataSavePath,featuresFilePath,liteFeaturesFilesPath,
             W2Vmodel = gensim.models.Word2Vec.load('Word2Vec.model'),D2V_200_model = gensim.models.Doc2Vec.load('Doc2Vec_200.model'),
             D2V_20_model = gensim.models.Doc2Vec.load('Doc2Vec_20.model')):

    speechDataSaveNames = os.listdir(speechDataSavePath)

    for i in speechDataSaveNames:
        if not i.startswith('.'):
            print(f'text prep: {i}')

            t1 = time.time()

            if '.tsv' in i:
                seperator = '\t'
            else:
                seperator = ','

            df = pd.read_csv(speechDataSavePath + "/" + i, sep=seperator)
            print(df.columns)

            print(f"starting d2v")
            docVec200 = create_Doc_vectors(df,D2V_200_model)
            docVec20 = create_Doc_vectors(df,D2V_20_model)

            print(f"starting w2v")
            wordVec200 = create_word_vectors(df['No_Stops_Transcript'],
                                            W2Vmodel)  # vectorizes the speeches using the word2vec model - text data




            sentimentVector = getSentiment(
                df['No_Stops_Transcript'])  # runs sentiment analysis and saves to the file path - text data

            print(f'Vectorized corpus: {i} in {(time.time() - t1) / 60} minutes')

            df['wordVec200'] = wordVec200
            df['docVec200'] = docVec200
            df['docVec20'] = docVec20
            df['vaderSent'] = sentimentVector["vaderSent"]
            df['blobSent'] = sentimentVector['blobSent']
            df_lite = df.drop(['No_Stops_Transcript', 'Transcript'], axis=1)

            fileName = i.split('(')[0]

            df_lite_name = f'{fileName}_lite_{str(df_lite.shape)}.tsv'
            df_lite.to_csv(
                liteFeaturesFilesPath + df_lite_name,
                sep='\t')

            df_name = f'{fileName}{str(df.shape)}.tsv'

            df.to_csv(
                featuresFilePath + df_name,
                sep='\t')

############# Setting up financial data and metadata #############
def setUpFinancialData():
    financialDataSavePath = '/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/GSPC.csv'
    downloadFinData(financialDataSavePath, '^GSPC', 1927, 12, 30, 23, 59)

    metadataFilePath = '/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/MetaData/'

    downloadFinData(metadataFilePath + 'USDX.csv', 'DX-Y.NYB', 1971, 1, 3, 23, 59)
    downloadFinData(metadataFilePath + 'BTC.csv', 'BTC-USD', 2014, 9, 16, 23, 59)
    downloadFinData(metadataFilePath + 'VIX.csv', '^VIX', 1990, 1, 1, 23, 59)
    downloadFinData(metadataFilePath + 'OIL.csv', 'CL%3DF', 2000, 8, 22, 23, 59)
    downloadFinData(metadataFilePath + 'NASDAQ_comp.csv', '%5EIXIC', 1971, 2, 4, 23, 59)
    downloadFinData(metadataFilePath + 'SSE_comp.csv', '000001.SS', 1997, 7, 1, 23, 59)

######run R files - GSPC_TS , metadata_setup and combining final data
def runR_files():
    R_call('GSPC_timeseries.R')
    R_call('metaDataSetup.R')
    R_call('combine_by_date.R')

###### Transpose vectorized speech variables and sentiment vectors
def runTranspose(full_df_filepath):
    full_df = pd.read_csv(full_df_filepath)


    full_df = transposeWordVectors(transposeSentiments(full_df),columnName='wordVec200')
    full_df = transposeDocVectors(full_df, columnName='docVec200',vecLen=200)
    full_df = transposeDocVectors(full_df, columnName='docVec20',vecLen=20)

    full_df_filepath = f'/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Complete_data/final_dataset{full_df.shape}.csv'

    full_df.to_csv(full_df_filepath)

#setUpFinancialData()
#runR_files()
#runTranspose()

saveFilePath = '/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Complete_data/final_dataset ( 73827 , 39 ) .csv'
runTranspose(saveFilePath)
#this script needs to:
#
# (1: download all the data, clean it, extract features (time series and text), and combine the data
# (2: do all the machine learning stuff
