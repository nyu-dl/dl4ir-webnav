'''
Convert article's text in the dataset to word embeddings using a pretrained word2vec dictionary.
'''
import h5py
import numpy as np
import nltk
import utils
import os
import parameters as prm
import time

def compute_idx(pages_path_in, pages_path_out, vocab):


    f = h5py.File(pages_path_in, 'r')

    if prm.att_doc and prm.att_segment_type == 'sentence':
        nltk.download('punkt')
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    os.remove(pages_path_out) if os.path.exists(pages_path_out) else None

    # Save to HDF5
    fout = h5py.File(pages_path_out,'a')

    if prm.att_doc:
        shape = (f['text'].shape[0],prm.max_segs_doc,prm.max_words)
    else:
        shape=(f['text'].shape[0],prm.max_words)

    idxs = fout.create_dataset('idx', shape=shape, dtype=np.int32)
    mask = fout.create_dataset('mask', shape=(f['text'].shape[0],), dtype=np.float32)

    i = 0
    for text in f['text']:
        st = time.time()

        if prm.att_doc:
            if prm.att_segment_type.lower() == 'section' or prm.att_segment_type.lower() == 'subsection':
                segs = ['']
                for line in text.split('\n'):
                    if prm.att_segment_type == 'section':
                        line = line.replace('===', '')
                    if line.strip().startswith('==') and line.strip().endswith('=='):
                        segs.append('')
                    segs[-1] += line.lower() + '\n'
            elif prm.att_segment_type.lower() == 'sentence':
                segs = tokenizer.tokenize(text.lower().decode('ascii', 'ignore'))
            else:
                raise ValueError('Not a valid value for the attention segment type (att_segment_type) parameter. Valid options are "section", "subsection" or "sentence".')

            segs = segs[:prm.max_segs_doc]
            idxs_, _ = utils.text2idx2(segs, vocab, prm.max_words)
            idxs[i,:len(idxs_),:] = idxs_
            mask[i] = len(idxs_)
        else:
            idx, _ = utils.text2idx2([text.lower()], vocab, prm.max_words)
            idxs[i,:] = idx[0]
        i += 1

        #if i > 3000:
        #    break

        print 'processing article', i, 'time', time.time()-st

    f.close()
    fout.close()
