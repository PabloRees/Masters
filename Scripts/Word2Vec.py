import os
import time
import gensim
import pandas as pd
import CyWord2Vec #not working yet

class MySentences(object):
    def __init__(self,dirname):
        self.dirname = dirname

    def __iter__(self):
        charDict = ["(", ")"]
        for fname in os.listdir(self.dirname): #reference: https://rare-technologies.com/word2vec-tutorial/
            if not fname.startswith("."):
                #print(fname+"____________")
                df = pd.read_csv(self.dirname + "/"+ fname,sep="\t", converters={'No Stops Transcript': pd.eval})
                for speech in df['No Stops Transcript']:
                    speechStr = ' '.join(speech)
                    sentenceList = speechStr.split(" . ")
                    for k in sentenceList:
                        yield k.split(" ")

#speechesList = list(df['No Stops Transcript'])

sentences = MySentences("/Users/pablo/Desktop/Masters /Github Repository/Masters/Sample data")

print(f"sentences: {sentences}")

#speechesList = [['first', 'sentence'], ['second', 'sentence'],['surely','there','are','senator', 'robinson', 'members', 'democratic', 'convention', 'friends', 'every', 'community', 'throughout',],['chairman', 'fellow', 'citizens', 'accepting', 'great', 'honor', 'brought', 'desire', 'speak', 'simply', 'plainly', 'every', 'man', 'woman', 'united']]
#print(speechesList)

t1 = time.time()
print(f"Start at {t1}")
model2 = CyWord2Vec.model(sentences)
print(f"Time taken (4 cores) = {time.time()-t1} seconds")

t1 = time.time()
print(f"Start at {t1}")
model = gensim.models.Word2Vec(sentences, min_count=5, vector_size=200) #builds the Word2Vec model, min_count refers to the min number of times a word appears in the corpus. Cector_size refers to the size of the output vector, alpha refers to the size of the gradient descent step
print(f"Time taken (1 core) = {time.time()-t1} seconds")




print(model.wv.index_to_key)

for index, word in enumerate(model.wv.index_to_key):
    if index == 10:
        break
    print(f"word #{index}/{len(model.wv.index_to_key)} is {word}")
