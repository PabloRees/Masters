import WebscrapeAndClean
import  SentimentAnalysis
import Word2Vec
import SP_500_download
speechDataSavePath = '/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Speech_data_test'
WebscrapeAndClean.run(speechDataSavePath)

Word2Vec_Model = Word2Vec.trainWord2Vec()

SentimentAnalysis()


#this script needs to:
#
# (1: download all the data, clean it, extract features (time series and text), combine the data and combine the data
# (2: do all the machine learning stuff