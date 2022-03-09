import os
from gensim.models import Word2Vec
import pandas as pd
import numpy as np

##Isolated run requirements
#loadFilePath = "/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Speech_data"
#saveFilePath = '/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Speech_data_with_vectors'
#model = trainWord2Vec("/Users/pablo/Desktop/Masters /Github Repository/Masters/Sample data")
#model.save('Word2Vec.model')

#creates a memory lazy iterator
class MySentences(object):
    def __init__(self,dirname):
        self.dirname = dirname

    def __iter__(self):
        charDict = ["(", ")"]

        for fname in os.listdir(self.dirname): #reference: https://rare-technologies.com/word2vec-tutorial/
            if not fname.startswith("."):
                print(fname+"____________")
                df = pd.read_csv(self.dirname + "/"+ fname,sep="\t", converters={'No Stops Transcript': pd.eval})
                for speech in df['No Stops Transcript']:
                    speechStr = ' '.join(speech)
                    sentenceList = speechStr.split(" . ")
                    for k in sentenceList:
                        yield k.split(" ") #creates an iterable list of every sentence in a speech, once one speech is exhausted, the next speech will be loaded into the iterable list of sentences.

def trainWord2Vec(dataFramesFilePath):
    sentences = MySentences(dataFramesFilePath)
    model = Word2Vec(sentences, min_count=5, vector_size=200, workers= 4, sg=1, compute_loss=True) #builds the Word2Vec model, min_count refers to the min number of times a word appears in the corpus. Vector_size refers to the size of the output vector, alpha refers to the size of the gradient descent step
    return model

#vectorizes the speeches using the mean of each of the vectors of each of the words in a speech
def vecSpeech_mean(speech,model):
    vectorList = [model.wv[word] for word in speech if word in model.wv.index_to_key]
    #print(vectorList)
    speechVec = np.mean(vectorList, axis=0)
    return speechVec

def create_vectors(speechList,model):
    speechVecList = []
    for speech in speechList:  # gets a speech from the 'no stops transcript'
        speechVec = vecSpeech_mean(speech,model)  # vectorizes the speech using the above function
        speechVecList.append(speechVec)  # appends the speechVector to a list
        print(f'{len(speechVecList)} speeches done')
    return speechVecList


