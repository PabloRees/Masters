import os
import gensim.models
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import concurrent.futures

vectorList = []
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
    #vectorList = [model.wv[word] for word in speech if word in model.wv.index_to_key] #could multithread this line because the order of the word vectors dont matter to the average

    vectorList.clear()
    speech = speech.replace(", '.'","")
    speech = speech[1:]
    speech = speech[1:]
    speech = speech[:-1]
    speech = speech[:-1]
    wordList = speech.split("', '")

    length = len(wordList)
    MAX_THREADS = 16
    if length == 0: length = 1

    threads = min(MAX_THREADS, length)

    with concurrent.futures.ThreadPoolExecutor(
            max_workers=threads) as executor:  # multithreading - its like 17 times faster than looping
        executor.map(getWordVec, wordList, chunksize=100)

    #print(vectorList)
    speechVec = np.mean(vectorList, axis=0)
    return speechVec

def create_vectors(speechList,model):
    speechVecList = []
    for speech in speechList:  # gets a speech from the 'no stops transcript'
        speechVec = vecSpeech_mean(speech,model)  # vectorizes the speech using the above function
        speechVecList.append(speechVec)  # appends the speechVector to a list

    return speechVecList

def getWordVec(word, model = gensim.models.Word2Vec.load('Word2Vec.model')):
    wordVec = []
    if word in model.wv.index_to_key:
        wordVec = model.wv[word]
        vectorList.append(wordVec)

    return wordVec

