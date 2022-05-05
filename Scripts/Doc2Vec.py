from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import gensim.models.doc2vec
import multiprocessing
import os
import pandas as pd
import numpy as np
from Transpose import transposeDocVectors

class MySpeeches(object):
    def __init__(self,dirname):
        self.dirname = dirname

    def __iter__(self):
        charDict = ["(", ")"]

        tag = 0
        for fname in os.listdir(self.dirname): #reference: https://rare-technologies.com/word2vec-tutorial/
            if not fname.startswith("."):
                print(fname+"____________")
                df = pd.read_csv(self.dirname + "/"+ fname,sep="\t", converters={'No Stops Transcript': pd.eval})

                for i in range(len(df['No Stops Transcript'])):
                    yield gensim.models.doc2vec.TaggedDocument(df['No Stops Transcript'].iloc[i], [f"{tag}/{i}/{df['Name'].iloc[i]}/{df['Type'].iloc[i]}]"])
                    tag+=1 #this line will run after the iterator has been called

def trainDoc2Vec(dataFramesFilePath):
    print('running trainDoc2Vec')
    cores = multiprocessing.cpu_count()
    assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"

    speeches = MySpeeches(dataFramesFilePath)

    doc2VecModel = Doc2Vec(dm=0, vector_size=200, negative=5, hs=0, min_count=2, sample=0,
                epochs=20, workers=cores)

    doc2VecModel.build_vocab(speeches)
    print('Vocab built')
    doc2VecModel.train(speeches, total_examples=doc2VecModel.corpus_count, epochs=doc2VecModel.epochs)
    print('Model trained')

    print(type(doc2VecModel))

    return doc2VecModel

def create_Doc_vectors(speechList,model= gensim.models.Doc2Vec.load('Doc2Vec.model')):

    docVecList=[]
    for document in speechList:  # gets a speech from the 'no stops transcript'
        if document in model.vocab:
            docVec = model.dv[document]
            docVecList.append(docVec)  # appends the speechVector to a list

    return docVecList

filePath = '/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Complete_data/heavy_final_dataset( 101566 , 21 ) .csv'
def getVecs(filePath=filePath,model = gensim.models.Doc2Vec.load('Doc2Vec.model')):

    df = pd.read_csv(filePath)
    docVecs = []
    for i in range(len(df['No.Stops.Transcript'])):
        docVecs.append(model.__getitem__(i))

    df['DocVecs'] = docVecs

    full_df = transposeDocVectors(df,'DocVecs')

    full_df.to_csv(filePath)

getVecs()




