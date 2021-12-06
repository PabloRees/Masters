import timeit
import CyWord2Vec

CyWord2Vec

cy = timeit.timeit('CyWord2Vec',setup='import CyWord2Vec',number=100)
py = timeit.timeit('Word2Vec',setup='import Word2Vec', number=100)


print(cy, py)
print(f"Cython is {py/cy}X faster than Python")

