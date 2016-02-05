'''
Custom theano class to access page links.
'''

import numpy as np
import theano
from theano import gof
from theano import tensor
import time


class Link(theano.Op):
    __props__ = ()

    def __init__(self, wiki, wikiemb, vocab, dim_emb):
        self.wiki = wiki
        self.wikiemb = wikiemb
        self.vocab = vocab
        self.dim_emb = dim_emb

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
        return theano.Apply(self, [x,x2,x3,x4], [tensor.ftensor3().type(), tensor.fmatrix().type(), \
                                           tensor.fmatrix().type(), tensor.ivector().type()])


    def perform(self, node, inputs, output_storage):
        #st = time.time()
        pages_id = inputs[0]
        p_truth = inputs[1]
        k_beam = inputs[2]
        it = inputs[3]

        max_link = 0
        links_all = []
        for i, page_id in enumerate(pages_id):
            if int(page_id) != -1:
                links = self.wiki.get_article_links(page_id)
                links_all.append(links)
                k = len(links)
                if k > max_link:
                    max_link = k
            else:
                links_all.append([])

        if max_link < k_beam:
            max_link = k_beam

        L = np.zeros((len(pages_id), max_link, self.dim_emb), np.float32)    
        L_m = np.zeros((len(pages_id), max_link), np.float32)
        l_page_id = -np.ones((len(pages_id), max_link), np.float32)
        l_truth = max_link * np.ones((len(pages_id)), np.int32) # initialize all elements with the stop action
        for i, links in enumerate(links_all):
            k = 0                
            # Get links' BoW.
            for link_id in links:
                L[i,k,:] = self.wikiemb.get_article_emb(link_id)
                L_m[i,k] = 1.0
                l_page_id[i,k] = link_id

                if link_id == p_truth[i]:
                    l_truth[i] = k

                k += 1

        output_storage[0][0] = L
        output_storage[1][0] = L_m
        output_storage[2][0] = l_page_id
        output_storage[3][0] = l_truth
        #print 'time Link op:', str(time.time() - st)
        #with open('out.log', "a") as fout:
        #    fout.write('it' + str(it) + 'time Link op:' + str(time.time() - st) + '\n')

    def grad(self, inputs, output_grads):
        return [tensor.zeros_like(ii, dtype=theano.config.floatX) for ii in inputs]

