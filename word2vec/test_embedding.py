from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import datapath
from pathlib import Path

model = KeyedVectors.load_word2vec_format(Path.cwd() / 'model_test.txt', binary=False)

print('dinner')
print(model.most_similar(positive='dinner'))
print('----------------')
print('wind')
print(model.most_similar(positive='wind'))
