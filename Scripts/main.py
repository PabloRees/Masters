import os

#from WebscrapeAndClean import *
import gensim.models

from SentimentAnalysis import *
from Word2Vec import *
from SP_500_download import *
from R_Interface import *
from Data_congregator import *

############# Setting up text data #############
speechDataSavePath = '/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Speech_data_test'

#runWSC(speechDataSavePath) #calls the WebScrapeAndClean process - text data

#Word2Vec_Model = trainWord2Vec(speechDataSavePath) #creates the word2vec model - text data

W2Vmodel = gensim.models.Word2Vec.load('Word2Vec.model')

def textPrep():
    for i in os.listdir(speechDataSavePath):
        if not i.startswith('.'):
            print(i)

            df = pd.read_csv(speechDataSavePath + "/" + i, sep='\t')
            speechesVector = create_vectors(df['No Stops Transcript'],
                                            W2Vmodel)  # vectorizes the speeches using the word2vec model - text data
            sentimentVector = getSentiment(
                df['No Stops Transcript'])  # runs sentiment analysis and saves to the file path - text data

            df['SpeechVectors'] = speechesVector
            df['vaderSent'] = sentimentVector["vaderSent"]
            df['blobSent'] = sentimentVector['blobSent']
            df_lite = df.drop(['No Stops Transcript', 'Transcript'], axis=1)

            fileName = i.split('(')[0]

            df_lite_name = fileName + str(df_lite.shape) + '.tsv'
            df_lite.to_csv(
                '/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Speech_data_lite/' + df_lite_name,
                sep='\t')

            df_name = fileName + str(df.shape) + '.tsv'
            df.to_csv(
                "/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Speech_data_with_features/" + df_name,
                sep='\t')

#textPrep()

############# Setting up financial data #############
#financialDataSavePath = '/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/GSPC_1.csv'
#downloadSP500(financialDataSavePath) #downloads and saves the S&P 500 data to the financialDataSavePath
#R_call('GSPC_timeseries.R')

#allSpeeches_df = combineSpeeches('/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Speech_data_lite')
R_call('combine_by_date.R')

full_df = pd.read_csv('final_dataset.csv')

print(full_df)

#this script needs to:
#
# (1: download all the data, clean it, extract features (time series and text), and combine the data
# (2: do all the machine learning stuff