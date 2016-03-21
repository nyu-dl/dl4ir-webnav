'''
Custom theano class to access page sentences.
'''
import numpy as np
import theano
from theano import gof
from theano import tensor
import utils
from nltk.tokenize import wordpunct_tokenize
import nltk
import time

class Sentence(theano.Op):
    __props__ = ()

    def __init__(self, wiki, vocab, n_consec):
        self.wiki = wiki
        self.vocab = vocab
        self.n_consec = n_consec # number of consecutive sections that are used to form a query
        nltk.download('punkt')
        self.tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    def make_node(self, x, x2, x3, x4):
        # check that the theano version has support for __props__.
        # This next line looks like it has a typo,
        # but it's actually a way to detect the theano version
        # is sufficiently recent to support the use of __props__.
        assert hasattr(self, '_props'), "Your version of theano is too old to support __props__."
        x = tensor.as_tensor_variable(x)
        x2 = tensor.as_tensor_variable(x2)
        x3 = tensor.as_tensor_variable(x3)
        x4 = tensor.as_tensor_variable(x4)
        return theano.Apply(self, [x, x2, x3, x4], [tensor.fvector().type(), tensor.ivector().type(), tensor.imatrix().type()])


    def perform(self, node, inputs, output_storage):
        #st = time.time()
        q = inputs[0]
        q_m = inputs[1]
        pages_id = inputs[2]
        div = inputs[3]

        R = np.zeros((len(pages_id)/div,), np.float32)
        best_pages_id = np.zeros((len(pages_id)/div,), np.int32)

        best_answers = []
        max_words = 0

        for i in range(0, len(pages_id), div):
            q_bow = {}
            for j, ax in enumerate(q[i/div]):
                if q_m[i/div][j] > 0.:
                    q_bow[ax] = 0
            set_q_bow = set(q_bow.keys())

            sents = []
            ref_id = []
            ref_range = []
            for j in range(div):
                page_id = pages_id[i+j]
                if int(page_id) != -1:
                    text = self.wiki.get_article_text(page_id)
                    sents_pre = self.tokenizer.tokenize(text.decode('ascii', 'ignore'))
                    n_consec = min(len(sents_pre),self.n_consec)
                    for sk in range(0,len(sents_pre)-n_consec+1):
                        sent = ''
                        for sj in range(n_consec):
                            sent += ' ' + sents_pre[sk+sj]
                        sents.append(sent.strip())
                        ref_id.append(page_id)

                    ref_range.append([j,len(sents)])
            s = np.zeros((len(sents)), np.float32)
            c = np.zeros((len(sents)), np.float32)
            sents_idx = []
            for j, sent in enumerate(sents):
                words = wordpunct_tokenize(sent.lower())
                sent_bow = {}
                for word in words:
                    if word in self.vocab:
                        sent_bow[self.vocab[word]] = 0
                sents_idx.append(words)
                c[j] = len(list(set(sent_bow.keys()) & set_q_bow)) # Count how many elements they have in common
                s[j] = len(sent_bow)
  
            match_rate = 2 * c / (len(set_q_bow) + s)
            idx = np.argmax(match_rate)
            R[i/div] = float(match_rate[idx] == 1.) # make reward \in {0,1}
            #R[i/div] = match_rate[idx] # make reward \in [0,1]
            best_pages_id[i/div] = ref_id[idx]
            sent_idx = utils.text2idx(sents_idx[idx], self.vocab)
            best_answers.append(sent_idx)
            if len(sent_idx) > max_words:
                max_words = len(sent_idx) 

        best_answers_ = -2*np.ones((len(best_answers), max_words), np.int32) #initialize with -2. -2 means stop word.
        for i, best_answer in enumerate(best_answers):
            best_answers_[i, :len(best_answer)] = best_answer 

        output_storage[0][0] = R
        output_storage[1][0] = best_pages_id
        output_storage[2][0] = best_answers_
        #with open('out.log', "a") as fout:
        #    fout.write('time Sentence op:' + str(time.time() - st) + '\n')

    def grad(self, inputs, output_grads):
        return [tensor.zeros_like(ii, dtype=theano.config.floatX) for ii in inputs]

