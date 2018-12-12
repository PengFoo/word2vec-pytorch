import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class SkipGramModel(nn.Module):
    """
    Skip-Gram model
    """
    def __init__(self, vocab_size: int, emb_dimension: int=200):
        super(SkipGramModel, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dimension = emb_dimension
        self.u_embeddings = nn.Embedding(vocab_size, emb_dimension, sparse=True)
        self.v_embeddings = nn.Embedding(vocab_size, emb_dimension, sparse=True)
        self.init_emb()

    def init_emb(self):
        """
        init the weight as original word2vec do.

        :return: None
        """
        initrange = 0.5 / self.emb_dimension
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(0, 0)

    def forward(self, pos_u, pos_v, neg_v):
        """
        forward process.
        the pos_u and pos_v shall be the same size.
        the neg_v shall be {negative_sampling_count} * size_of_pos_u
        eg:
        5 sample per batch with 200d word embedding and 6 times neg sampling.
        pos_u 5 * 200
        pos_v 5 * 200
        neg_v 5 * 6 * 200

        :param pos_u:  positive pairs u, list
        :param pos_v:  positive pairs v, list
        :param neg_v:  negative pairs v, list
        :return:
        """
        emb_u = self.u_embeddings(pos_u)  # batch_size * emb_size
        emb_v = self.v_embeddings(pos_v)  # batch_size * emb_size
        emb_neg = self.v_embeddings(neg_v)  # batch_size * neg sample size * emb_size

        pos_score = torch.mul(emb_u, emb_v).squeeze()
        pos_score = torch.sum(pos_score, dim=1)
        pos_score = F.logsigmoid(pos_score)

        neg_score = torch.bmm(emb_neg, emb_u.unsqueeze(2)).squeeze()
        neg_score = F.logsigmoid(-neg_score)

        return -1 * (torch.sum(pos_score) + torch.sum(neg_score))

    def save_embedding(self, id2word: dict, file_name: str='word_vectors.txt', use_cuda: bool=False):
        """
        Save all embeddings to file.
        As this class only record word id, so the map from id to word has to be transfered from outside.

        :param id2word: map from word id to word.
        :param file_name: file name.
        :param use_cuda:
        :return:
        """
        if use_cuda:
            embedding = self.u_embeddings.weight.cpu().data.numpy()
        else:
            embedding = self.u_embeddings.weight.data.numpy()
        fout = open(file_name, 'w', encoding='utf-8')
        fout.write('%d %d\n' % (len(id2word), self.emb_dimension))
        for wid, w in id2word.items():
            e = embedding[wid]
            e = ' '.join(map(lambda x: str(x), e))
            fout.write('%s %s\n' % (w, e))


def test():
    vocab_size, embed_size = 10, 100
    model = SkipGramModel(vocab_size, embed_size)
    id2word = dict()
    for i in range(vocab_size):
        id2word[i] = str(i)
    # test model saving
    model.save_embedding(id2word)


if __name__ == '__main__':
    test()
