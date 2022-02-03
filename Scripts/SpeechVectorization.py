import pandas as pd
import numpy as np
import os
from gensim.models import Word2Vec

model = Word2Vec.load('Word2Vec.model') #this is a huge corpus (trained on my speeches) linking words to word vectors
#print(model.wv.index_to_key) #prints the vocabulary in the model

#vectorizes the speeches using the mean of each of the vectors of each of the words in a speech
def vecSpeech_mean(speech):

    vectorList = [model.wv[word] for word in speech if word in model.wv.index_to_key]
    #print(vectorList)
    speechVec = np.mean(vectorList, axis=0)

    return speechVec

loadFilePath = "/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Speech_data"
saveFilePath = '/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Speech_data_with_vectors'
for i in os.listdir(loadFilePath):#gets all the corpuses
    if not i.startswith('.'):
        print(f'Starting to vectorize {i}')
        df = pd.read_csv(loadFilePath+'/'+i, sep='\t')#loads a speech type corpus into df
        speechVecList = []
        for speech in df['No Stops Transcript']:#gets a speech from the 'no stops transcript'
            speechVec = vecSpeech_mean(speech)#vectorizes the speech using the above function
            speechVecList.append(speechVec)#appends the speechVector to a list
            print(f'{len(speechVecList)} speeches done')
        df['SpeechVec_mean'] = speechVecList#adds the list to the speech type corpus df
        df.to_csv(saveFilePath+'/'+i, sep='\t')
        print(f'{i} done')





