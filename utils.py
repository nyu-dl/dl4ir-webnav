'''
Miscellaneous functions.
'''

import numpy as np
import cPickle as pkl
from nltk.tokenize import wordpunct_tokenize
import re

def BOW(words, vocab):
    '''
    Convert a list of words to the BoW representation.
    '''
    bow = {} # BoW densely represented as <vocab word idx: quantity>
    for word in words:
        if word in vocab:
            if vocab[word] not in bow:
                bow[vocab[word]] = 0.
            bow[vocab[word]] += 1.

    bow_v = np.asarray(bow.values())
    sumw = float(bow_v.sum())
    if sumw == 0.:
        sumw = 1.
    bow_v /= sumw

    return [bow.keys(), bow_v]


def BOW2(texts, vocab, dim):
    '''
    Convert a list of texts to the BoW dense representation.
    '''
    out = np.zeros((len(texts), dim), dtype=np.int32)
    mask = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        bow = BOW(wordpunct_tokenize(text), vocab)
        out[i,:len(bow[0])] = bow[0]
        mask[i,:len(bow[1])] = bow[1]

    return out, mask


def text2idx(words, vocab):
    '''
    Convert a list of words to their corresponding vocabulary indexes.
    '''
    idxs = []
    for word in words:
        if word in vocab:
            idxs.append(vocab[word])
        else:
            idxs.append(-1)

    return idxs


def idx2text(idxs, vocabinv):
    '''
    Convert list of vocabulary indexes to text.
    '''
    out = []
    for i in idxs:
        if i >= 0:
            out.append(vocabinv[i])
        elif i == -1:
            out.append('<UNK>')
        else:
            break

    return " ".join(out)


def n_words(words, vocab):
    '''
    Counts the number of words that have an entry in the vocabulary.
    '''
    c = 0
    for word in words:
        if word in vocab:
            c += 1
    return c


def load_vocab(path, n_words=None):
    dic = pkl.load(open(path, "rb"))
    vocab = {}

    if not n_words:
        n_words = len(dic.keys())

    for i, word in enumerate(dic.keys()[:n_words]):
        vocab[word] = i
    return vocab


def compute_tf(words, vocab):
    '''
    Compute the term frequency in the document and return a dictionary of <word, freq>
    '''
    out = {}
    for word in words:
        if word in vocab:
            if word not in out:
                out[word] = 0.
            out[word] += 1.

    return out





