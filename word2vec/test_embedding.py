from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import datapath
from pathlib import Path

model = KeyedVectors.load_word2vec_format(Path.cwd() / 'model_test.txt', binary=False)

print('女孩')
print(model.most_similar(positive='女孩'))
print('----------------')
print('中国')
print(model.most_similar(positive='中国'))
