from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import gensim.models.doc2vec
import multiprocessing
import os
import pandas as pd
import numpy as np
from Transpose import transposeDocVectors

class MySpeeches(object):
    def __init__(self,dirname,combine_sameday_speeches):

        if '.csv' in dirname or '.tsv' in dirname:
            dirname = dirname.split('/')[:-1]
            dirname = ''.join([f'{item}/'for item in dirname])

        print(dirname)

        self.dirname = dirname
        self.combine_sameday_speeches = combine_sameday_speeches

    def __iter__(self):
        charDict = ["(", ")"]

        for fname in os.listdir(self.dirname): #reference: https://rare-technologies.com/word2vec-tutorial/
            if not fname.startswith("."):
                print(fname+"____________")

                if '.csv' in fname:
                    seperator = ','
                elif '.tsv' in fname:
                    seperator = "\t"

                df = pd.read_csv(self.dirname + "/"+ fname,sep= seperator)

                print('df loaded')

                if self.combine_sameday_speeches:
                    for i in range(len(df['No_Stops_Transcript'])):
                        yield gensim.models.doc2vec.TaggedDocument(df['No_Stops_Transcript'].iloc[i], [f"{df['Date'].iloc[i]}"]) #DONT REMOVE SQUARE BRACKETS ON TAGS!!!
                else:
                    for i in range(len(df['No_Stops_Transcript'])):
                        yield gensim.models.doc2vec.TaggedDocument(df['No_Stops_Transcript'].iloc[i], [f"{i}_{df['Date'].iloc[i]}_{df['Name'].iloc[i]}_{df['Type'].iloc[i]}"])

def trainDoc2Vec(dataFramesFilePath,vecSize,model_type,combine_sameday_speeches:bool):
    print(f'running trainDoc2Vec for {model_type} model')
    cores = multiprocessing.cpu_count()
    assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"

    speeches = MySpeeches(dataFramesFilePath,combine_sameday_speeches)

    if not model_type in ['PV_DBOW','PV_DM']:
        raise ValueError('Model type must be PV_DBOW or PV_DM')

    if model_type == 'PV_DBOW':
        doc2VecModel = Doc2Vec(dm=0, vector_size=vecSize, negative=5, hs=0, min_count=10, sample=0,
                epochs=10, workers=cores)

    else: #model_type = 'PV_DM'
        doc2VecModel = Doc2Vec(dm=1, vector_size=vecSize,window = 5, dm_mean=1, negative=5, hs=0, min_count=10, sample=0,
                epochs=10, workers=cores)

    doc2VecModel.build_vocab(speeches)
    print('Vocab built')
    doc2VecModel.train(speeches, total_examples=doc2VecModel.corpus_count, epochs=doc2VecModel.epochs)
    print('Model trained')

    if combine_sameday_speeches:combined = 'combined'
    else: combined = ''

    print(f'{type(doc2VecModel)} : {model_type}')
    doc2VecModel.save(f'Doc2Vec_{vecSize}_{combined}_{model_type}.model')

    return doc2VecModel

def create_Doc_vectors(df,model,combine_sameday_speeches:bool):
    docVecList=[]
    count = 0

    for i in range(len(df)):

        if combine_sameday_speeches :
            tag = f"{df['Date'].iloc[i]}" #DON'T ADD SQUARE BRACKETS TO TAG!!!
        else:
            tag = f"{i}_{df['Date'].iloc[i]}_{df['Name'].iloc[i]}_{df['Type'].iloc[i]}"

        if tag in model.dv.index_to_key:
            docVec = model.dv[tag]
            print(type(docVec))
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







