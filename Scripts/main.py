import os
from WebscrapeAndClean import *
import gensim.models
from SentimentAnalysis import *
from Word2Vec import *
from fin_data_downloader import *
from R_Interface import *
from Data_congregator import *

############# Setting up text data #############
speechDataSavePath = '/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Speech_data'

#runWSC(speechDataSavePath) #calls the WebScrapeAndClean process - text data

#Word2Vec_Model = trainWord2Vec(speechDataSavePath) #creates the word2vec model - text data

featuresFilePath = "/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Speech_data_with_features"
liteFeaturesFilesPath = '/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Speech_data_lite/'

def textPrep(speechDataSaveName):

    W2Vmodel = gensim.models.Word2Vec.load('Word2Vec.model')

    if not speechDataSaveName.startswith('.'):
        print(f'text prep: {speechDataSaveName}')

        t1 = time.time()

        df = pd.read_csv('/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Speech_data' + "/" + speechDataSaveName, sep='\t')
        speechesVector = create_vectors(df['No Stops Transcript'],
                                            W2Vmodel)  # vectorizes the speeches using the word2vec model - text data
        sentimentVector = getSentiment(df['No Stops Transcript'])  # runs sentiment analysis and saves to the file path - text data

        print(f'Vectorized corpus: {speechDataSaveName} in {(time.time() - t1)/60} minutes')

        df['SpeechVectors'] = speechesVector
        df['vaderSent'] = sentimentVector["vaderSent"]
        df['blobSent'] = sentimentVector['blobSent']
        df_lite = df.drop(['No Stops Transcript', 'Transcript'], axis=1)

        fileName = speechDataSaveName.split('(')[0]

        df_lite_name = fileName + str(df_lite.shape) + '.tsv'
        df_lite.to_csv(
                '/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Speech_data_lite/' + df_lite_name,
                sep='\t')

        df_name = fileName + str(df.shape) + '.tsv'
        df.to_csv(
                "/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Speech_data_with_features" + df_name,
                sep='\t')

speechDataSaveNames = os.listdir(speechDataSavePath)
speechDataSaveNames.pop(0)
speechDataSaveNames.pop(0)

MAX_THREADS = 16
length = len(speechDataSaveNames)
if length == 0: length = 1

threads = min(MAX_THREADS, length)

with concurrent.futures.ThreadPoolExecutor(
            max_workers=threads) as executor:  # multithreading - its like 17 times faster than looping
        executor.map(textPrep, speechDataSaveNames)

#textPrep(speechDataSavePath, featuresFilePath, liteFeaturesFilesPath)

combinedLiteSpeeches = combineSpeeches(liteFeaturesFilesPath)

completeDataFilePath = '/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Complete_data'

combinedLiteSpeeches.to_csv(completeDataFilePath + '/combinedLiteSpeeches.tsv', sep='\t')

############# Setting up financial data #############
financialDataSavePath = '/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/GSPC.csv'
downloadFinData(financialDataSavePath,'^GSPC',1927, 12, 30, 23, 59)

############ Download Financial Metadata ##############
metadataFilePath = '/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/MetaData/'

downloadFinData(metadataFilePath + 'USDX.csv','DX-Y.NYB',1971, 1, 3, 23, 59)
downloadFinData(metadataFilePath + 'BTC.csv','BTC-USD',2014, 9, 16, 23, 59)
downloadFinData(metadataFilePath + 'VIX.csv','^VIX',1990, 1, 1, 23, 59)
downloadFinData(metadataFilePath + 'OIL.csv','CL%3DF',2000, 8, 22, 23, 59)
downloadFinData(metadataFilePath + 'NASDAQ_comp.csv','%5EIXIC',1971, 2, 4, 23, 59)
downloadFinData(metadataFilePath + 'SSE_comp.csv','000001.SS',1997, 7, 1, 23, 59)

######run R files - GSPC_TS , metadata_setup and combining final data
R_call('GSPC_timeseries.R')
R_call('metaDataSetup.R')
R_call('combine_by_date.R')


#this script needs to:
#
# (1: download all the data, clean it, extract features (time series and text), and combine the data
# (2: do all the machine learning stuff