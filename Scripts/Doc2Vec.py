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

        for fname in os.listdir(self.dirname): #reference: https://rare-technologies.com/word2vec-tutorial/
            if not fname.startswith("."):
                print(fname+"____________")

                if '.csv' in fname:
                    seperator = ','
                elif '.tsv' in fname:
                    seperator = "\t"

                df = pd.read_csv(self.dirname + "/"+ fname,sep= seperator, converters={'No_Stops_Transcript': pd.eval})

                for i in range(len(df['No_Stops_Transcript'])):
                    yield gensim.models.doc2vec.TaggedDocument(df['No_Stops_Transcript'].iloc[i], [f"{i}_{df['Date'].iloc[i]}_{df['Name'].iloc[i]}_{df['Type'].iloc[i]}]"])

def trainDoc2Vec(dataFramesFilePath,vecSize):
    print('running trainDoc2Vec')
    cores = multiprocessing.cpu_count()
    assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"

    speeches = MySpeeches(dataFramesFilePath)

    doc2VecModel = Doc2Vec(dm=0, vector_size=vecSize, negative=5, hs=0, min_count=2, sample=0,
                epochs=20, workers=cores)

    doc2VecModel.build_vocab(speeches)
    print('Vocab built')
    doc2VecModel.train(speeches, total_examples=doc2VecModel.corpus_count, epochs=doc2VecModel.epochs)
    print('Model trained')

    print(type(doc2VecModel))
    doc2VecModel.save(f'Doc2Vec_{vecSize}.model')

    return doc2VecModel

def create_Doc_vectors(df,model):

    docVecList=[]
    count = 0
    for i in range(len(df)):
        tag = f"{i}_{df['Date'].iloc[i]}_{df['Name'].iloc[i]}_{df['Type'].iloc[i]}]"


        if tag in model.docvecs.index_to_key:
            #print(f"{i} of {len(df)}")
            docVec = model.dv[tag]
            docVecList.append(docVec)  # appends the speechVector to a list
            count+=1

    print(f'{count} of {len(df)} tags matched')
    return docVecList

filePath = '/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Complete_data/heavy_final_dataset( 101566 , 21 ) .csv'
def getVecs(filePath=filePath,model = gensim.models.Doc2Vec.load('Doc2Vec.model')):

    df = pd.read_csv(filePath)
    docVecs = []
    for i in range(len(df['No_Stops_Transcript'])):
        docVecs.append(model.__getitem__(i))

    df['DocVecs'] = docVecs

    full_df = transposeDocVectors(df,'DocVecs')

    full_df.to_csv(filePath)





