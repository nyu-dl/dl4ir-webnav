'''
Convert article's text in the dataset to word embeddings using a pretrained word2vec dictionary.
'''
import h5py
import numpy as np
from nltk.tokenize import wordpunct_tokenize
import nltk
import utils
import cPickle as pkl
import os
import parameters as prm
import time

def compute_emb(pages_path_in, pages_path_out, vocab):

    wemb = pkl.load(open(prm.wordemb_path, 'rb'))
    dim_emb = wemb[wemb.keys()[0]].shape[0]
    W = 0.01 * np.random.randn(len(vocab), dim_emb).astype(np.float32)
    for word, pos in vocab.items():
        if word in wemb:
            W[pos,:] = wemb[word]

    f = h5py.File(pages_path_in, 'r')

    if prm.att_doc and prm.att_segment_type == 'sentence':
        nltk.download('punkt')
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    os.remove(pages_path_out) if os.path.exists(pages_path_out) else None

    # Save to HDF5
    fout = h5py.File(pages_path_out,'a')

    if prm.att_doc:
        shape = (f['text'].shape[0],prm.max_segs_doc,prm.dim_emb)
    else:
        shape=(f['text'].shape[0],prm.dim_emb)

    embs = fout.create_dataset('emb', shape=shape, dtype=np.float32)
    mask = fout.create_dataset('mask', shape=(f['text'].shape[0],), dtype=np.float32)

    i = 0
    for text in f['text']:
        st = time.time()

        if prm.att_doc:
            if prm.att_segment_type == 'section':
                segs = ['']
                for line in text.split('\n'):
                    if line.strip().startswith('==') and line.strip().endswith('=='):
                        segs.append('')
                    segs[-1] += line + '\n'
            elif prm.att_segment_type == 'sentence':
                segs = tokenizer.tokenize(text.decode('ascii', 'ignore'))
            else:
                raise ValueError('Not a valid value for the attention segment type (att_segment_type) parameter.')

            segs = segs[:prm.max_segs_doc]
            emb_ = utils.Word2Vec_encode(segs, wemb)
            embs[i,:len(emb_),:] = emb_
            mask[i] = len(emb_)
        else:
            bow0, bow1 = utils.BOW(wordpunct_tokenize(text.lower()), vocab)
            emb = (W[bow0] * bow1[:,None]).sum(0)
            embs[i,:] = emb
        i += 1
        #if i > 3000:
        #    break

        print 'processing article', i, 'time', time.time()-st

    f.close()
    fout.close()
