from gensim.models import Word2Vec
import TSNE_scatter_plot as tsne

model_loaded = Word2Vec.load('Word2Vec.model')

print(model_loaded.wv.most_similar(positive=['covid']))
print(model_loaded.wv.most_similar(positive=['wuhan']))

print(model_loaded.wv.doesnt_match(['wuhan', 'obama', 'michelle']))
print(model_loaded.wv.doesnt_match(['covid', 'market', 'iraq']))

#print(model_loaded.wv.similarity('covid', 'market'))
#print(model_loaded.wv.similarity('covid', 'crash'))
#print(model_loaded.wv.similarity('covid', 'iraq'))
#print(model_loaded.wv.similarity('covid', 'terror'))
#print(model_loaded.wv.similarity('covid', 'trump'))
#print(model_loaded.wv.similarity('covid', 'election'))

print(model_loaded.wv.most_similar(positive=["britain", "paris"], negative=["france"], topn=3)) #which word is to A as B is to C?

wordForPlot = 'mandela'  #choose word to produce tsne scatter plot built from Word2Vec model
#red word is input, blue words are 10 most similar, green words are 10 least similar
#interestting words include 'marijuana', 'mexican', 'michelle', 'vaccine', 'isis', 'trudeau', 'military', 'transgender', antifa','recovery'

tsne.tsnescatterplot(model_loaded, wordForPlot, [i[0] for i in model_loaded.wv.most_similar(negative=[wordForPlot])])