'''
Miscellaneous functions.
'''

import numpy as np
import cPickle as pkl
from nltk.tokenize import wordpunct_tokenize
import parameters as prm
from random import randint

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


def Word2Vec_encode(texts, wemb):
    
    out = np.zeros((len(texts), prm.dim_emb), dtype=np.float32)
    for i, text in enumerate(texts):
        words = wordpunct_tokenize(text)
        n = 0.
        for word in words:
            if word in wemb:
                out[i,:] += wemb[word]
                n += 1.
        out[i,:] /= max(1.,n)

    return out


def text2idx2(texts, vocab, dim):
    '''
    Convert a list of texts to their corresponding vocabulary indexes.
    '''
    out = -np.ones((len(texts), dim), dtype=np.int32)
    mask = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        j = 0
        for word in wordpunct_tokenize(text):
            if word in vocab:
                out[i,j] = vocab[word]
                mask[i,j] = 1.
                j += 1

                if j == dim:
                    break

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


def load_synonyms():
    dic_thes = {}
    with open(prm.path_thes_dat, 'rb') as f:
        data = f.read().lower()
    header = 0
    with open(prm.path_thes_idx, 'rb') as f:
        for line in f:
            if header < 2:
                header += 1
                continue
            word_idx = line.rstrip().split("|")
            word, idx = word_idx[0], word_idx[1]
            idx = int(idx)
            j=0
            desc = ""
            while data[idx+j] != "\n":
                desc += data[idx+j]
                j += 1
            word_numlines = desc.rstrip().split("|")
            word, numlines = word_numlines[0], word_numlines[1]
            numlines = int(numlines)
            dic_thes[word] = []
            k = 0
            desc = ""
            while True:
                j += 1 
                desc += data[idx+j]
                if data[idx+j] == "\n":
                    k += 1
                    synonyms = desc.rstrip().split("|")[1:] #do not consider the first word because it refers to the POS tagging
                    dic_thes[word].extend(synonyms) #extend list of synonyms
                    desc = "" #start a new line
                if k == numlines:
                    break
    return dic_thes


def augment(texts, dic_thes):
    if prm.aug<2:
        return texts

    out = []
    for text in texts:

        words_orig = wordpunct_tokenize(text)
        maxrep = max(2,int(0.1*len(words_orig))) #define how many words will be replaced. For now, leave the maximum number as 10% of the words
        
        for j in range(prm.aug):
            words = list(words_orig) #copy
            for k in range(randint(1,maxrep)):
                idx = randint(0,len(words)-1)
                word = words[idx]
                if word in dic_thes:
                    
                    synonym = min(np.random.geometric(0.5), len(dic_thes[word])-1) #chose the synonym based on a geometric distribution
                    #print 'fp',fp,"word", word,"synonym",dic_thes[word][synonym]
                    words[idx] = dic_thes[word][synonym]

            out.append(" ".join(words))

    return out     
