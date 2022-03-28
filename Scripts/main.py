import os
from WebscrapeAndClean import *
import gensim.models
from SentimentAnalysis import *
from Word2Vec import *
from fin_data_downloader import *
from R_Interface import *
from Data_congregator import *
from Transpose import *

def textPrep(speechDataSavePath,featuresFilePath,liteFeaturesFilesPath):

    W2Vmodel = gensim.models.Word2Vec.load('Word2Vec.model')

    speechDataSaveNames = os.listdir(speechDataSavePath)

    for i in speechDataSaveNames:
        if not i.startswith('.'):
            print(f'text prep: {i}')

            t1 = time.time()

            df = pd.read_csv(speechDataSavePath + "/" + i, sep='\t')
            speechesVector = create_vectors(df['No Stops Transcript'],
                                            W2Vmodel)  # vectorizes the speeches using the word2vec model - text data
            sentimentVector = getSentiment(
                df['No Stops Transcript'])  # runs sentiment analysis and saves to the file path - text data

            print(f'Vectorized corpus: {i} in {(time.time() - t1) / 60} minutes')

            df['SpeechVectors'] = speechesVector
            df['vaderSent'] = sentimentVector["vaderSent"]
            df['blobSent'] = sentimentVector['blobSent']
            df_lite = df.drop(['No Stops Transcript', 'Transcript'], axis=1)

            fileName = i.split('(')[0]

            df_lite_name = fileName + str(df_lite.shape) + '.tsv'
            df_lite.to_csv(
                liteFeaturesFilesPath + df_lite_name,
                sep='\t')

            df_name = fileName + str(df.shape) + '.tsv'
            df.to_csv(
                featuresFilePath + df_name,
                sep='\t')

############# Setting up text data #############
def setUpTextData():
    speechDataSavePath = '/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Speech_data'

    runWSC(speechDataSavePath)  # calls the WebScrapeAndClean process - text data

    Word2Vec_Model = trainWord2Vec(speechDataSavePath)  # creates the word2vec model - text data

    featuresFilePath = "/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Speech_data_with_features/"
    liteFeaturesFilesPath = '/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Speech_data_lite/'

    textPrep(speechDataSavePath, featuresFilePath, liteFeaturesFilesPath)

    completeDataFilePath = '/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Complete_data'

    combinedLiteSpeeches = combineSpeeches(liteFeaturesFilesPath)
    combinedLiteSpeeches.to_csv(completeDataFilePath + '/combinedLiteSpeeches.tsv', sep='\t')

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
def runTranspose():
    full_df_filepath = '/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Complete_data/final_dataset.csv ( 106774 , 35 ) .csv'
    full_df = pd.read_csv(full_df_filepath)
    full_df = transposeVectors(transposeSentiments(full_df))

    full_df_filepath = full_df_filepath.split('Complete_data/')[0] + 'Complete_data/final_dataset' + str(full_df.shape) + '.csv'

    full_df.to_csv(full_df_filepath)

#setUpTextData()
#setUpFinancialData()
#runR_files()
runTranspose()

#this script needs to:
#
# (1: download all the data, clean it, extract features (time series and text), and combine the data
# (2: do all the machine learning stuff