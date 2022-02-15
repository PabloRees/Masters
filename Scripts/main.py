from WebscrapeAndClean import *
from SentimentAnalysis import *
from Word2Vec import *
from SpeechVectorization import *
from SP_500_download import *
from R_Interface import *
from Data_congregator import *

############# Setting up text data #############
speechDataSavePath = '/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Speech_data_test'

runWSC(speechDataSavePath) #calls the WebScrapeAndClean process - text data

Word2Vec_Model = trainWord2Vec(speechDataSavePath) #creates the word2vec model - text data

create_vectors(speechDataSavePath,speechDataSavePath) #vectorizes the speeches using the word2vec model - text data

runSentAnal(speechDataSavePath,speechDataSavePath) #runs sentiment analysis and saves to the file path - text data


############# Setting up financial data #############
financialDataSavePath = '/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/GSPC.csv'
downloadSP500(financialDataSavePath) #downloads and saves the S&P 500 data to the financialDataSavePath
R_call('R_call_test.R')


#this script needs to:
#
# (1: download all the data, clean it, extract features (time series and text), and combine the data
# (2: do all the machine learning stuff