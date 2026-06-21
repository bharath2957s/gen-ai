import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import gensim.downloader as api
from scipy.spatial.distance import cosine

model = api.load("glove-wiki-gigaword-100")

print("Vector of 'king':")
print(model['king'][:5])

print("\nSimilar words to 'king':")
print(model.most_similar('king'))

print("\nking - man + woman = ?")
print(model.most_similar(positive=['king', 'woman'], negative=['man']))

sim = 1 - cosine(model['king'], model['queen'])
print("\nSimilarity between king and queen:", sim)
