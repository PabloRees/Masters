import os
from gensim.models import Word2Vec
import pandas as pd
import cython

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
                        yield k.split(" ")

def trainWord2Vec(dataFramesFilePath):
    sentences = MySentences(dataFramesFilePath)
    #model = CyWord2Vec.model(sentences)
    model = Word2Vec(sentences, min_count=5, vector_size=200, workers= 4, sg=1, compute_loss=True) #builds the Word2Vec model, min_count refers to the min number of times a word appears in the corpus. Cector_size refers to the size of the output vector, alpha refers to the size of the gradient descent step
    return model

model = trainWord2Vec("/Users/pablo/Desktop/Masters /Github Repository/Masters/Sample data")

#model.save('Word2Vec.model')

#model_loaded = Word2Vec.load('Word2Vec.model')


#print(model_loaded.wv['president'])

#print(model_loaded.wv.s)