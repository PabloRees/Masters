import os
import pandas as pd
from gensim.models import Word2Vec
from sklearn.decomposition import PCA

WordVecModel = Word2Vec.load('Word2Vec.model')

print(WordVecModel)