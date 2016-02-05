'''
Convert article's text in the dataset to word embeddings using a pretrained word2vec dictionary.
'''

import h5py
import numpy as np
from nltk.tokenize import wordpunct_tokenize
import utils
import cPickle as pkl
import os
import parameters as prm

def compute_emb(pages_path_in, pages_path_out, vocab):

    wemb = pkl.load(open(prm.wordemb_path, 'rb'))
    dim_emb = wemb[wemb.keys()[0]].shape[0]
    W = 0.01 * np.random.randn(len(vocab), dim_emb).astype(np.float32)
    for word, pos in vocab.items():
        if word in wemb:
            W[pos,:] = wemb[word]

    f = h5py.File(pages_path_in, 'r')
    i = 0
    embs = []
    for text in f['text']:
        #print 'processing article', i
        bow0, bow1 = utils.BOW(wordpunct_tokenize(text.lower()), vocab)
        emb = (W[bow0] * bow1[:,None]).sum(0)
        embs.append(emb)
        i += 1
    f.close()

    # Save to HDF5
    os.remove(pages_path_out) if os.path.exists(pages_path_out) else None
    fout = h5py.File(pages_path_out,'a')
    fout.create_dataset('emb', data=np.asarray(embs))
    fout.close()
