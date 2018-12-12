# Word2vec in Pytorch

> This repo has learnt a lot from [this repo](https://github.com/Adoni/word2vec_pytorch)

This repo implements the **SkipGram model with negative sampling** of Word2vec by Mikolov.

Tricks below are also implemented:
- subsampling
- negative sampling with pow weight decay
- learning rate decay

# Requirements
- PyTorch >= 0.4.1
- Gensim >= 3.6.0 (for testing only)

# Fast run
To quickly run the train model, just run 

` python train.py `

 which uses a Chinese corpus to train the Word2vec model. There is another toy corpus in English you can use located in `data/trainset.txt`, which is actually a "Jane Eyre" novel.

Issues and PRs are welcomed!

# Reference
[1] Mikolov T, Sutskever I, Chen K, et al. Distributed representations of words and phrases and their compositionality[C]//Advances in neural information processing systems. 2013: 3111-3119.