from gensim.models import Word2Vec

def model(sentences):
    print("CyWord2Vec functioning...barely")
    model = Word2Vec(sentences, min_count=5, vector_size=200, workers= 4, sg=1, compute_loss=True) #builds the Word2Vec model, min_count refers to the min number of times a word appears in the corpus. Cector_size refers to the size of the output vector, alpha refers to the size of the gradient descent step


