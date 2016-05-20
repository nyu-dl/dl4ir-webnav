'''
Simple inverted-indexing algorithm,
'''
import parameters as prm
import wiki
import numpy as np
import cPickle as pkl
from nltk.tokenize import wordpunct_tokenize
import time
from collections import OrderedDict


def get_candidates(qatp):

    print 'loading data...'
    idf = pkl.load(open(prm.idf_path, "rb"))
    wk = wiki.Wiki(prm.pages_path)

    print 'creating vocabulary...'
    vocab = {}
    for q,_,_,_ in qatp:
        words = wordpunct_tokenize(q.lower())
        for word in words:
            if word in idf:
                vocab[word] = {}


    print 'creating inverted index...'
    i = 0
    for text in wk.get_text_iter():
        if i%10000==0:
            print 'article', i
        words = wordpunct_tokenize(text.lower())
        for word in words:
            if word in vocab:
                vocab[word][i] = 0

        #if i > 500000:
        #    break
        i += 1

    print 'selecting pages...'
    candidates = []
    for i,[q,_,_,_] in enumerate(qatp):
        st = time.time()
        words = wordpunct_tokenize(q.lower())
        scores = {}

        for word in words:
            if word in vocab:
                if len(vocab[word]) < 100000:
                    for pageid in vocab[word].keys(): 
                        if pageid not in scores:
                            scores[pageid] = 0.
                        scores[pageid] += idf[word]
        idxs = np.argsort(np.asarray(scores.values()))[::-1]

        pages = scores.keys()

        if len(idxs)==0:
            print 'error question:', q

        c = OrderedDict()
        for idx in idxs[:prm.max_candidates]:
            c[pages[idx]] = 0

        candidates.append(c)
        print 'sample ' + str(i) + ' time ' + str(time.time()-st)

        #if i > 10000:
        #    break

    return candidates

