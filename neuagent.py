'''
Build and run the RNN model
'''
import cPickle as pkl
import time
import numpy as np
import theano
import theano.tensor as tensor
from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from collections import OrderedDict, deque
import utils
from op_link import Link
from op_sentence import Sentence
from sklearn.decomposition import PCA
import wiki
import qp
import parameters as prm
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab since the server might not have an X server.
import matplotlib.pyplot as plt
from nltk.tokenize import wordpunct_tokenize
import copy
import itertools
import random

# compute_test_value is 'off' by default, meaning this feature is inactive
#theano.config.compute_test_value = 'warn' # Use 'warn' to activate this feature

# Set the random number generators' seeds for consistency
SEED = 123
np.random.seed(SEED)

def vis_att(pages_idx, query, alpha, wiki, vocab, idx):
    rows = [prm.root_page.title()]
    for pageidx in pages_idx[:-1]:
        if pageidx != -1:
            rows.append(wiki.get_article_title(pageidx).decode('utf-8', 'ignore').title())
        else:
            break
            #rows.append('Stop')

    rows = rows[::-1]

    columns = []
    for word in wordpunct_tokenize(query):
        if word.lower() in vocab:
            columns.append(str(word))
    columns = columns[:prm.max_words_query*prm.n_consec]

    alpha = alpha[:len(rows),:len(columns)]
    alpha = alpha[::-1]

    fig,ax=plt.subplots(figsize=(27,10))
    #Advance color controls
    norm = matplotlib.colors.Normalize(0,1)
    im = ax.pcolor(alpha,cmap=plt.cm.gray,edgecolors='w',norm=norm)
    fig.colorbar(im)
    ax.set_xticks(np.arange(0,len(columns))+0.5)
    ax.set_yticks(np.arange(0,len(rows))+0.5)
    ax.tick_params(axis='x', which='minor', pad=15)
    # Here we position the tick labels for x and y axis
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    ax.axis('tight') # correcting pyplot bug that add extra white columns.
    plt.xticks(rotation=90)
    fig.subplots_adjust(bottom=0.2)
    fig.subplots_adjust(left=0.2)
    #Values against each labels
    ax.set_xticklabels(columns,minor=False,fontsize=18)
    ax.set_yticklabels(rows,minor=False,fontsize=18)
    plt.savefig('vis' + str(idx) + '.svg')
    plt.close()

def np_floatX(data):
    return np.asarray(data, dtype=config.floatX)


def _slice(_x, n, dim):
    if _x.ndim == 3:
        return _x[:, :, n * dim:(n + 1) * dim]
    return _x[:, n * dim:(n + 1) * dim]


def get_minibatches_idx(n, minibatch_size, shuffle=False, max_samples=None):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    if max_samples:
        idx_list = idx_list[:max_samples]
        n = max_samples

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params


def dropout_layer(state_before, is_train, trng):
    proj = tensor.switch(is_train,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=(1-prm.dropout), n=1,
                                        dtype=state_before.dtype)),
                         state_before * (1-prm.dropout))
    return proj


def load_params(path, params):
    pp = np.load(path)
    for kk, vv in params.iteritems():
        if kk in pp:
            if params[kk].shape == pp[kk].shape:
                params[kk] = pp[kk]
            else:
                print 'The shape of layer', kk, params[kk].shape, 'is different from shape of the stored layer with the same name', pp[kk].shape, '.'
        else:
            print '%s is not in the archive' % kk

    return params


def load_wemb(params, vocab):
    wemb = pkl.load(open(prm.wordemb_path, 'rb'))
    dim_emb_orig = wemb.values()[0].shape[0]

    W = 0.01 * np.random.randn(prm.n_words, dim_emb_orig).astype(config.floatX)
    for word, pos in vocab.items():
        if word in wemb:
            W[pos,:] = wemb[word]
    
    if prm.dim_emb < dim_emb_orig:
        pca =PCA(n_components=prm.dim_emb, copy=False, whiten=True)
        W = pca.fit_transform(W)

    params['W'] = W

    return params


def itemlist(tparams):
    return [vv for kk, vv in tparams.iteritems()]


def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(config.floatX)


def matrix(dim):
    return np.concatenate([ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim)], axis=1)


def softmax_mask(x, mask):
    m = tensor.max(x, axis=-1, keepdims=True)
    e_x = tensor.exp(x - m) * mask
    return e_x / tensor.maximum(e_x.sum(axis=-1, keepdims=True), 1e-8) #this small constant avoids possible division by zero created by the mask


def init_params():
    params = OrderedDict()

    params['l_a_init'] = 0.01 * np.random.randn(prm.dim_emb,).astype(config.floatX) # initial values
    params['h_init'] = 0.01 * np.random.randn(prm.n_rnn_layers, prm.dim_proj).astype(config.floatX) # initial values
    params['c_init'] = 0.01 * np.random.randn(prm.n_rnn_layers, prm.dim_proj).astype(config.floatX) # initial values

    if prm.encoder.lower() == 'lstm':
        mul = 4
    else:
        mul = 1

    params['E_L'] = 0.01 * np.random.randn(prm.dim_emb, mul * prm.dim_proj).astype(config.floatX) # document
    params['E_Q'] = 0.01 * np.random.randn(prm.dim_emb, mul * prm.dim_proj).astype(config.floatX) # query
    params['U_I'] = 0.01 * np.random.randn(prm.dim_proj, mul * prm.dim_proj).astype(config.floatX) # hiddent state t-1
    params['b'] = np.zeros((mul * prm.dim_proj,)).astype(config.floatX) # bias

    for i in range(1, prm.n_rnn_layers):
        i = str(i)
        params['E_L'+i] = 0.01 * np.random.randn(prm.dim_emb, mul * prm.dim_proj).astype(config.floatX) # document
        params['E_Q'+i] = 0.01 * np.random.randn(prm.dim_emb, mul * prm.dim_proj).astype(config.floatX) # query
        params['U_H'+i] = 0.01 * np.random.randn(prm.dim_proj, mul * prm.dim_proj).astype(config.floatX) # hidden state t-1
        params['U_I'+i] = 0.01 * np.random.randn(prm.dim_proj, mul * prm.dim_proj).astype(config.floatX) # hidden state n-1
        params['b'+i] = np.zeros((mul * prm.dim_proj,)).astype(config.floatX) # bias

    params['stop'] = 0.01 * np.random.randn(prm.dim_emb).astype(config.floatX) # stop action vector
    params['U_O'] = 0.01 * np.random.randn(prm.dim_proj, prm.dim_proj).astype(config.floatX) # score
    params['b_U_O'] = np.zeros((prm.dim_proj,)).astype(config.floatX) # bias

    for i in range(prm.n_doc_layers_nav):
        if i == 0:
            i = ''
            in_dim = prm.dim_emb
        else:
            in_dim = prm.dim_proj
        params['U_L' + str(i)] = 0.01 * np.random.randn(in_dim, prm.dim_proj).astype(config.floatX) # doc embedding
        params['b_U_L' + str(i)] = np.zeros((prm.dim_proj,)).astype(config.floatX) # bias

    ns = [prm.dim_proj] + prm.scoring_layers_nav + [1]
    for i in range(len(ns)-1):
        if i == 0:
            i_ = ''
        else:
            i_ = str(i+1)  # +1 for compatibility purposes.
        params['U_R'+i_] = 0.01 * np.random.randn(ns[i], ns[i+1]).astype(config.floatX) # score    
        params['b_U_R'+i_] = np.zeros((ns[i+1],)).astype(config.floatX) # bias

    if prm.att_query:

        n_features = [prm.dim_emb,] + prm.filters_query
        for i in range(len(prm.filters_query)):
            params['Ww_att_q'+str(i)] = 0.001 * np.random.randn(n_features[i+1], n_features[i], 1, prm.window_query[i]).astype(config.floatX)
            params['bw_att_q'+str(i)] = np.zeros((n_features[i+1],)).astype(config.floatX) # bias score

        q_feat_size = n_features[-1]

        params['Wq_att_q'] = 0.001 * np.random.randn(q_feat_size, prm.dim_proj).astype(config.floatX) # query
        params['Wh_att_q'] = 0.001 * np.random.randn(prm.dim_proj, prm.dim_proj).astype(config.floatX) # hidden state
        params['Wl_att_q'] = 0.001 * np.random.randn(prm.dim_emb, prm.dim_proj).astype(config.floatX) # link embedding
        params['bq_att_q'] = np.zeros((prm.dim_proj,)).astype(config.floatX) # bias
        params['We_att_q'] = 0.001 * np.random.randn(prm.dim_proj, 1).astype(config.floatX) # score
        params['be_att_q'] = np.zeros((1,)).astype(config.floatX) # bias score

    if prm.att_doc:

        n_features = [prm.dim_emb,] + prm.filters_doc
        for i in range(len(prm.filters_doc)):
            params['Ww_att_d'+str(i)] = 0.01 * np.random.randn(n_features[i+1], n_features[i], 1, prm.window_doc[i]).astype(config.floatX)
            params['bw_att_d'+str(i)] = np.zeros((n_features[i+1],)).astype(config.floatX) # bias score

        doc_feat_size = n_features[-1]

        params['Wq_att_d'] = 0.01 * np.random.randn(prm.dim_emb, prm.dim_proj).astype(config.floatX) # query
        params['Wh_att_d'] = 0.01 * np.random.randn(prm.dim_proj, prm.dim_proj).astype(config.floatX) # hidden state
        params['Wl_att_d'] = 0.01 * np.random.randn(doc_feat_size, prm.dim_proj).astype(config.floatX) # link embedding
        params['bq_att_d'] = np.zeros((prm.dim_proj,)).astype(config.floatX) # bias
        params['We_att_d'] = 0.01 * np.random.randn(prm.dim_proj, 1).astype(config.floatX) # score
        params['be_att_d'] = np.zeros((1,)).astype(config.floatX) # bias score

    if prm.learning.lower() == 'reinforce' and prm.idb:
        params['R_W'] = 0.01 * np.random.randn(prm.dim_proj, 1).astype(config.floatX) # question   
        params['R_b'] = np.zeros((1,)).astype(config.floatX) # bias

    params['W'] = 0.01 * np.random.randn(prm.n_words, prm.dim_emb).astype(config.floatX) # vocab to word embeddings
    params['UNK'] = 0.01 * np.random.randn(1, prm.dim_emb).astype(config.floatX) # vector for UNK words

    exclude_params = {}
    if prm.fixed_wemb:
        exclude_params['W'] = True

    return params, exclude_params


def rnn_layer(x, h_, c_, m_):
    
    if prm.encoder.lower() == 'lstm':
        i = tensor.nnet.sigmoid(_slice(x, 0, prm.dim_proj))
        f = tensor.nnet.sigmoid(_slice(x, 1, prm.dim_proj))
        o = tensor.nnet.sigmoid(_slice(x, 2, prm.dim_proj))
        c = tensor.tanh(_slice(x, 3, prm.dim_proj))
    
        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_
    
        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_
    else:
        c = c_
        h = tensor.tanh(x) * m_[:, None]
    return h, c


def val(q_a, q_m, h_, l_a_, c_, m_, L_a, L_m, tparams_v, tparams, k_beam, n_samples, uidx, is_train, trng):
    
    def fparams(name):
        return tparams_v[tparams.keys().index(name)]

    n_links = L_a.shape[1] + 1
    
    if prm.att_query:
       
        # Convolution
        q_aw = q_a.dimshuffle(0, 2, 'x', 1) # (n_samples, dim_emb, 1, n_words)
        for j in range(len(prm.filters_query)):
            q_aw = tensor.nnet.conv2d(q_aw,
                                        fparams('Ww_att_q'+str(j)),
                                        border_mode=(0, prm.window_query[j]//2))
            q_aw += fparams('bw_att_q'+str(j))[None,:,None,None]
            q_aw = tensor.maximum(q_aw, 0.)
            #q_aw = tensor.nnet.relu(q_aw) # relu results in NAN. Use maximum() instead.

        q_aw = q_aw[:, :, 0, :].dimshuffle(0, 2, 1)
        
        e = tensor.dot(q_aw, fparams('Wq_att_q'))
        e += tensor.dot(h_[-1], fparams('Wh_att_q'))[:,None,:]
        e += tensor.dot(l_a_, fparams('Wl_att_q'))[:,None,:]
        e += fparams('bq_att_q')
        e = tensor.tanh(e)
        e = tensor.dot(e, fparams('We_att_q')) + fparams('be_att_q')
        e = e.reshape((e.shape[0],e.shape[1]))

        # repeat for beam search
        q_m_ = tensor.extra_ops.repeat(q_m, k_beam, axis=0)
        alpha = softmax_mask(e, q_m_)
        q_at = (alpha[:,:,None] * q_a).sum(1)
    else:
        alpha = tensor.alloc(np.array(0., dtype=np.float32), q_a.shape[0], q_a.shape[1])
        q_at = q_a

    alpha_q = alpha

    h = tensor.zeros_like(h_)
    c = tensor.zeros_like(c_)

    # Multi-layer lstm
    for i in range(prm.n_rnn_layers):

        i_ = '' if i == 0 else str(i)

        a = tensor.dot(q_at, fparams('E_Q' + i_))
        if prm.dropout > 0:
            a = dropout_layer(a, is_train, trng)

        b = tensor.dot(l_a_, fparams('E_L' + i_))
        if prm.dropout > 0:
            b = dropout_layer(b, is_train, trng)

        preact = a + b
        preact += tensor.dot(h_[i], fparams('U_I' + i_))
        preact += fparams('b' + i_)

        if i > 0:
            hp = tensor.dot(h[i-1], fparams('U_H' + i_))
            if prm.dropout > 0:
                hp = dropout_layer(hp, is_train, trng)
            preact += hp

        h_i, c_i = rnn_layer(preact, h_[i], c_[i], tensor.neq(m_,-1.).astype('float32'))
        h = tensor.set_subtensor(h[i], h_i)
        c = tensor.set_subtensor(c[i], c_i)

    if prm.att_doc:

        # Convolution.
        L_aw = L_a.reshape((L_a.shape[0] * L_a.shape[1], L_a.shape[2], L_a.shape[3]))
        L_aw = L_aw.dimshuffle(0, 2, 'x', 1) # (n_samples*n_docs, n_emb, 1, n_char)
        for j in range(len(prm.filters_doc)):
            L_aw = tensor.nnet.conv2d(L_aw,
                                        fparams('Ww_att_d'+str(j)),
                                        border_mode=(0,  prm.window_doc[j]//2))
            L_aw += fparams('bw_att_d'+str(j))[None,:,None,None]
            L_aw = tensor.maximum(L_aw, 0.)
            # L_aw = tensor.nnet.relu(L_aw) # relu results in NAN. Use maximum() instead.

        L_aw = L_aw[:, :, 0, :].dimshuffle(0, 2, 1)
        L_aw = L_aw.reshape((L_a.shape[0], L_a.shape[1], L_a.shape[2], L_aw.shape[2]))

        e = tensor.dot(L_aw, fparams('Wl_att_d'))
        e += tensor.dot(h[-1], fparams('Wh_att_d'))[:,None,None,:]
        e += tensor.dot(q_at, fparams('Wq_att_d'))[:,None,None,:]
        e += fparams('bq_att_d')
        e = tensor.tanh(e)
        e = tensor.dot(e, fparams('We_att_d')) + fparams('be_att_d')
        e = e.reshape((e.shape[0],e.shape[1],e.shape[2]))
        
        alpha = softmax_mask(e, L_m)
        L_at = (alpha[:,:,:,None] * L_a).sum(2)
        L_m = L_m.any(2).astype('float32') 
    else:
        L_at = L_a

    # Append stop action
    stop = fparams('stop')[None, None, :]
    stop = tensor.extra_ops.repeat(x=stop, repeats=n_samples * k_beam, axis=0)
    L_as = tensor.concatenate([stop, L_at], axis=1)
    stop_m = tensor.alloc(np_floatX(1.), n_samples * k_beam, 1)
    L_ms = tensor.concatenate([stop_m, L_m], axis=1)

    z = tensor.tanh(tensor.dot(h[-1], fparams('U_O')) + fparams('b_U_O'))

    L_as2 = L_as
    for i in range(prm.n_doc_layers_nav):
        if i == 0:
            i_ = ''
        else:
            i_ = str(i)

        L_as2 = tensor.dot(L_as2, fparams('U_L'+i_)) + fparams('b_U_L'+i_)
        if prm.dropout > 0:
            L_as2 = dropout_layer(L_as2, is_train, trng)
        L_as2 = tensor.tanh(L_as2)
    
    res = tensor.dot(L_as2 * z[:,None,:], fparams('U_R')) + fparams('b_U_R')

    for i in range(1,len(prm.scoring_layers_nav)+1):

        if prm.dropout > 0:
            res = dropout_layer(res, is_train, trng)

        res = tensor.tanh(res) # tanh here instead after the dot product makes no tanh in the last layer.
        res = tensor.dot(res, fparams('U_R'+str(i+1))) + fparams('b_U_R'+str(i+1))

    res = res.reshape((n_samples, k_beam * n_links)) # Reshape for beam search
    L_ms = L_ms.reshape((n_samples, k_beam * n_links))

    score = res * L_ms

    return score, h, c, L_as, L_ms, alpha_q


def adam(lr0, tparams, grads, iin, out, updates):
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k) for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    
    f_grad_shared = theano.function(iin, out, updates=gsup+updates, \
                                    on_unused_input='ignore', allow_input_downcast=True)

    b1 = 0.1
    b2 = 0.001
    e = 1e-8

    updates = []

    i = theano.shared(np.float32(0.))
    i_t = i + 1.
    fix1 = 1. - b1**(i_t)
    fix2 = 1. - b2**(i_t)
    lr_t = lr0 * (tensor.sqrt(fix2) / fix1)

    for p, g in zip(tparams.values(), gshared):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * tensor.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (tensor.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))

    f_update = theano.function([lr0], [], updates=updates, on_unused_input='ignore')

    return f_grad_shared, f_update


def compute_emb(x, W):

    def _step(xi, emb, W):
        if prm.att_doc:
            new_shape = (xi.shape[0], xi.shape[1], xi.shape[2], prm.dim_emb)
        else:
            new_shape = (xi.shape[0], xi.shape[1], prm.dim_emb)

        out = W[xi.flatten()].reshape(new_shape).sum(-2)
        return out / tensor.maximum(1., tensor.neq(xi,-1).astype('float32').sum(-1, keepdims=True))

    if prm.att_doc:
        emb_init = tensor.alloc(0., x.shape[1], x.shape[2], prm.dim_emb)
    else:
        emb_init = tensor.alloc(0., x.shape[1], prm.dim_emb)

    (embs), scan_updates = theano.scan(_step,
                                sequences=[x],
                                outputs_info=[emb_init],
                                non_sequences=[W],
                                name='emb_scan',
                                n_steps=x.shape[0])

    return embs


def ff(q, q_m, k_beam, trng, is_train, options, uidx, tparams, mixer, sup, root_pages, max_hops, acts_p, rl_idx=None, get_links=None):

    n_samples = q.shape[0]
    off = 1e-8
    epsilon = tensor.maximum(prm.epsilon_min, prm.epsilon_start - (prm.epsilon_start - prm.epsilon_min) * (uidx / prm.epsilon_decay))

    if not get_links:
        get_links = Link(options['wiki'], options['wikipre'], options['vocab']) # instantiate custom function to get wiki links

    # append vector for UNK words (index == -1).
    W_ = tensor.concatenate([tparams['W'], tparams['UNK']], axis=0)

    def _step(it, act_p, p_, m_, l_a_, h_, c_, q_a, pr_all, W_, k_beam, uidx, is_train, sup, mixer, *tparams_v):

        L_a, L_m, l_page, l_truth = get_links(p_, act_p, it, uidx, k_beam)

        if prm.compute_emb:
            L_a = compute_emb(L_a, W_)
        dist, h, c, L_as, L_ms, alpha_q = val(q_a, q_m, h_, l_a_, c_, m_, L_a, L_m, tparams_v, tparams, k_beam, n_samples, uidx, is_train, trng)

        n_links = L_as.shape[1]

        if prm.learning.lower() == 'q-learning':

            if rl_idx: #if this is the replay memory pass, just use the q-value function
                dist = tensor.nnet.sigmoid(dist) * L_ms
                res_ = dist.argmax(1)

            else: # otherwise, select actions using epsilon-greedy or softmax.

                if prm.act_sel.lower() == 'epsilon-greedy':
                    dist = tensor.nnet.sigmoid(dist) * L_ms

                    greedy = tensor.eq(is_train,1.).astype('float32') * (trng.uniform(size=(n_samples,)) > epsilon) \
                           + tensor.eq(is_train,0.).astype('float32')

                    randd = tensor.floor(trng.uniform(size=(n_samples,)) * L_ms.sum(1)).astype('int32')

                    res_pre = tensor.eq(it, 0.).astype('int32') * dist[:,:n_links].argsort(axis=1)[:,::-1][:, :k_beam].flatten().astype("int32") \
                            + tensor.neq(it, 0.).astype('int32') * dist.argsort(axis=1)[:,::-1][:, :k_beam].reshape((n_samples * k_beam,)).astype("int32")

                    # Repeat for beam search
                    greedy = tensor.extra_ops.repeat(greedy, k_beam, axis=0)
                    randd = tensor.extra_ops.repeat(randd, k_beam, axis=0)

                    res_ = (1. - greedy) * randd + greedy * res_pre

                elif prm.act_sel.lower() == 'softmax':
                    dist = softmax_mask(dist, L_ms)

                    # if training, sample. Otherwise, use the maximum value.
                    lp_ = tensor.eq(is_train,1.).astype('float32') * trng.multinomial(n=1, pvals=dist, dtype=dist.dtype) \
                        + tensor.eq(is_train,0.).astype('float32') * dist
                    
                    res_ = tensor.eq(it, 0.).astype('int32') * lp_[:,:n_links].argsort(axis=1)[:,::-1][:, :k_beam].flatten().astype("int32") \
                         + tensor.neq(it, 0.).astype('int32') * lp_.argsort(axis=1)[:,::-1][:, :k_beam].reshape((n_samples * k_beam,)).astype("int32")

        else:
            dist = softmax_mask(dist, L_ms)

            lp_ = tensor.eq(is_train,1.).astype('float32') * trng.multinomial(n=1, pvals=dist, dtype=dist.dtype) \
                + tensor.eq(is_train,0.).astype('float32') * \
                  (tensor.log(pr_all[:it] + off).sum(0)[:,None] + tensor.log(dist.reshape((n_samples*k_beam,-1)) + off)).reshape((n_samples,-1))

            res_ = tensor.eq(it, 0.).astype('int32') * lp_[:,:n_links].argsort(axis=1)[:,::-1][:, :k_beam].flatten().astype("int32") \
                 + tensor.neq(it, 0.).astype('int32') * lp_.argsort(axis=1)[:,::-1][:, :k_beam].reshape((n_samples * k_beam,)).astype("int32")

        # Select action: supervised, RL, or mixed.
        if prm.mixer > 0 and prm.learning.lower() == 'reinforce':
            # Mixed
            l_idx = ((it < mixer) * l_truth + (1 - (it < mixer)) * res_).astype("int32")
        else: 
            # Supervised or RL
            if rl_idx: #if this is the replay forward pass, just choose the same action taken in the past
                l_idx = rl_idx[:,it]
            else: # Otherwise, use the supervised signal or the action chosen by the policy.
                l_idx = (sup * l_truth + (1 - sup) * res_).astype("int32")
       
        l_idx0 = (k_beam * tensor.floor(tensor.arange(l_idx.shape[0]) / k_beam)  + tensor.floor(l_idx / (n_links)) ).astype('int32')
        l_idx1 = tensor.mod(l_idx, n_links).astype('int32')

        l_a = L_as[l_idx0, l_idx1, :]

        dist = dist.reshape((n_samples*k_beam, n_links))
        l_prob = dist[l_idx0, l_idx1] # get the probability of the chosen action.
        l_ent = -(dist * tensor.log(dist + off)).sum(1) # get the entropy.
        pr_all = tensor.set_subtensor(pr_all[it], l_prob)

        # supervised only: compute the cost for page selection
        cost_p = -tensor.log(dist[tensor.arange(dist.shape[0]), l_truth] + off)

        # check if the stop action was chosen, and
        # mark the sample as "not stop" by storing the current iteration.
        m = tensor.neq(l_idx1, 0).astype("float32")
        m = m * it - (1. - m)
        m = m.astype('float32')

        # Get indices of the next articles.
        p = l_page[l_idx0, l_idx1]

        # the returned variable in the scan function must have same size in all iterations.
        dist_ = tensor.alloc(0., n_samples * k_beam, prm.max_links+1)        
        dist_ = tensor.set_subtensor(dist_[tensor.arange(n_samples*k_beam), :dist.shape[1]], dist)

        # the returned variable in the scan function must have same size in all iterations.
        l_page_ = tensor.alloc(-1, n_samples * k_beam, prm.max_links+1).astype('int32')
        l_page_ = tensor.set_subtensor(l_page_[tensor.arange(n_samples*k_beam), :l_page.shape[1]], l_page)

        return p, m, l_a, h, c, l_prob, l_ent, cost_p, l_idx, dist_, alpha_q, l_page_

    #get embeddings for the queries
    q_a = W_[q.flatten()].reshape((q.shape[0], q.shape[1], prm.dim_emb)) * q_m[:,:,None]

    if not prm.att_query:
        q_a = q_a.sum(1) / tensor.maximum(1., q_m.sum(1, keepdims=True))

    #repeat question for beam search
    q_a = tensor.extra_ops.repeat(q_a, k_beam, axis=0)
    root_pages_ = tensor.extra_ops.repeat(root_pages, k_beam)

    l_a_init = tensor.extra_ops.repeat(tparams['l_a_init'][None,:], k_beam * n_samples, axis=0)
    h_init = tensor.extra_ops.repeat(tparams['h_init'][:,None,:], k_beam * n_samples, axis=1)
    c_init = tensor.extra_ops.repeat(tparams['c_init'][:,None,:], k_beam * n_samples, axis=1)

    pr_all = tensor.alloc(1., max_hops+1, k_beam * n_samples)

    (pages_idx, mask, l_a, h, _, l_prob, l_ent, cost_p, l_idx, dist, alpha_q, l_page), scan_updates = theano.scan(_step,
                                sequences=[tensor.arange(max_hops+1), acts_p],
                                outputs_info=[root_pages_, #page idx
                                              tensor.alloc(0., k_beam * n_samples),  # mask
                                              l_a_init,
                                              h_init,
                                              c_init,
                                              None, # l_prob
                                              None,  # l_ent
                                              None,  # cost_p
                                              None,  # l_idx
                                              None,  # dist
                                              None,  # alpha_q
                                              None,  # l_page
                                              ],
                                non_sequences=[q_a, pr_all, W_, k_beam, uidx, is_train, sup, mixer]+tparams.values(),
                                name='lstm_layers',
                                n_steps=max_hops+1,
                                strict=True)

    #convert mask
    mask = mask.max(0)
    indices = tensor.repeat(tensor.arange(max_hops+1)[:,None], mask.shape[0], axis=1)
    mask = (indices <= mask[None,:]).astype('float32')

    return (pages_idx, mask, l_a, h[:,-1,:,:], l_prob, l_ent, cost_p, root_pages_, l_idx, dist, alpha_q, l_page), scan_updates, get_links


def build_model(tparams, tparams_next, baseline_vars, options):
    trng = RandomStreams(SEED)
    off = 1e-8  # small constant to avoid log 0 = -inf
    consider_constant = []

    is_train = theano.shared(np_floatX(0.)) # Used for dropout.
    mixer = theano.shared(np.asarray(0, dtype=np.int32)) # Used for MIXER.
    sup = theano.shared(np_floatX(0.)) # Supervised or not
    max_hops = theano.shared(np.asarray(prm.max_hops_pred, dtype=np.int32)) # Max number of iterations
    k_beam = theano.shared(np.asarray(prm.k, dtype=np.int32)) # top-k items in the beam search.
    
    q = tensor.imatrix('q')
    q_m = tensor.fmatrix('q_m')
    root_pages = tensor.fvector('root_pages')
    acts_p = tensor.imatrix('acts_p')

    #used only when prm.learning = 'q-learning'
    uidx = tensor.iscalar('uidx')
    rs_q = tensor.imatrix('rs_q')
    rs_q_m = tensor.fmatrix('rs_q_m')
    rl_idx = tensor.imatrix('rl_idx')
    rt = tensor.fmatrix('rt')
    rR = tensor.fmatrix('rR')

    """
    q.tag.test_value = np.zeros((prm.batch_size_train,prm.n_consec*prm.max_words_query), dtype='int32')
    q_m.tag.test_value = np.ones((prm.batch_size_train,prm.n_consec*prm.max_words_query), dtype=theano.config.floatX)
    root_pages.tag.test_value = np.zeros((prm.batch_size_train,), dtype=theano.config.floatX)
    acts_p.tag.test_value = np.zeros((prm.max_hops_train+1,prm.batch_size_train), dtype='int32')
    uidx.tag.test_value = np.zeros((1,), dtype='int32')
    rs_q_a.tag.test_value = np.zeros((prm.batch_size_train,prm.dim_emb), dtype=theano.config.floatX)
    rs_q_m.tag.test_value = np.zeros((prm.batch_size_train,prm.n_consec*prm.max_words_query), dtype=theano.config.floatX)
    rl_idx.tag.test_value = np.zeros((prm.batch_size_train,), dtype='int32')
    rt.tag.test_value = np.zeros((prm.batch_size_train,), dtype=theano.config.floatX)
    rR.tag.test_value = np.zeros((prm.batch_size_train,), dtype=theano.config.floatX)
    """
        
    (pages_idx, mask, l_a, h, l_prob, l_ent, cost_p, root_pages_, l_idx, dist, alpha_q, l_page), scan_updates_a, _ = \
        ff(q, q_m, k_beam, trng, is_train, options, uidx, tparams, mixer, sup, root_pages, max_hops, acts_p)

    # Get only the used probabilities.
    mask_ = tensor.concatenate([tensor.alloc(np_floatX(1.), 1, mask.shape[1]), mask], axis=0)[:-1,:]
    l_prob *= mask_   # l_prob.shape = (n_iterations, n_samples)
    l_ent *= mask_   # l_ent.shape = (n_iterations, n_samples)

    get_sent = Sentence(options['wiki'], options['vocab'], prm.n_consec) # instantiate custom function to get sentences 
    pages_idx_ = tensor.concatenate([root_pages_[None,:], pages_idx[:-1]], axis=0)

    # get last valid action before the stop action. In case the all the mask is True, get the last action.
    js = (tensor.minimum(mask.shape[0] - 1, mask.sum(axis=0))).astype("int32") 
    sel_docs = pages_idx_[js, tensor.arange(js.shape[0])]

    R, best_answer = get_sent(q, q_m, sel_docs, k_beam)

    # in case the agent didn't stop (all mask is true), the reward is zero.
    R *= tensor.neq(mask.sum(0), mask.shape[0]).astype('float32').reshape((R.shape[0], k_beam)).any(1)

    l_aT = l_a.dimshuffle((1,0,2))
    l_aT = l_aT.reshape((q.shape[0],-1, prm.dim_emb))

    sel_docs = sel_docs.reshape((-1, k_beam))

    # the first doc always has the best prob.
    best_doc = sel_docs[:, 0]

    f_pred = theano.function([q, q_m, root_pages, acts_p, uidx], \
                             [best_doc, best_answer, R, pages_idx, sel_docs, js, dist, alpha_q, l_page], \
                              updates=scan_updates_a, name='f_pred', on_unused_input='ignore')

    # entropy regularization
    cost_ent = -prm.erate * l_ent
    
    if prm.learning.lower() == 'supervised':
        # cost for link selection.
        cost = ((cost_p + cost_ent) * mask_).sum(0).mean()

        # costs for document scoring.
        a = tensor.neq(acts_p,-1).astype('int32').sum(0) - 1

        baseline_updates = []

    elif prm.learning.lower() == 'q-learning':

        (_, m, _, _, _, _, _, _, _, _, q_vals), scan_updates_b, get_links = \
                ff(rs_q, rs_q_m, k_beam, trng, is_train, \
                    options, uidx, tparams, mixer, sup, \
                    root_pages, max_hops, acts_p, rl_idx)

        m = m.T
        m_ = tensor.concatenate([tensor.alloc(np_floatX(1.), m.shape[0], 1), m], axis=1)[:,:-1]

        q_vals = q_vals.dimshuffle((1,0,2))
        
        if prm.update_freq > 1:
            (_, _, _, _, _, _, _, _, _, _, n_q_vals), scan_updates_c, _ = \
                ff(rs_q, rs_q_m, k_beam, trng, is_train, \
                    options, uidx, tparams_next, mixer, sup, \
                    root_pages, max_hops, acts_p, rl_idx, get_links)

            n_q_vals = n_q_vals.dimshuffle((1,0,2))
            # left shift n_q_vals and add zeros at the end.
            n_q_vals = tensor.concatenate([n_q_vals[:,1:,:], tensor.zeros_like(n_q_vals[:,0,:])[:,None,:]], axis=1)

        else:
            # left shift n_q_vals and add zeros at the end.
            n_q_vals = tensor.concatenate([q_vals[:,1:,:], tensor.zeros_like(q_vals[:,0,:])[:,None,:]], axis=1)
            n_q_vals *= tensor.ones_like(n_q_vals) # Dummy operation

            # Don't update weights with respect to n_q_vals
            n_q_vals = theano.gradient.disconnected_grad(n_q_vals)


        q_vals_ = q_vals.reshape((-1, q_vals.shape[2]))
        n_q_vals_ = n_q_vals.reshape((-1,n_q_vals.shape[2]))
        rR_ = rR.flatten()
        rt_ = rt.flatten()
        rl_idx_ = rl_idx.flatten()

        target = rR_ + (tensor.ones_like(rt_) - rt_) * prm.discount * n_q_vals_.max(1)
        diff = target - q_vals_[tensor.arange(rl_idx_.shape[0]), rl_idx_]

        if prm.clip > 0.:
            # If we simply take the squared clipped diff as our loss,
            # then the gradient will be zero whenever the diff exceeds
            # the clip bounds. To avoid this, we extend the loss
            # linearly past the clip point to keep the gradient constant
            # in that regime.
            # 
            # This is equivalent to declaring d loss/d q_vals to be
            # equal to the clipped diff, then backpropagating from
            # there, which is what the DeepMind implementation does.
            quadratic_part = tensor.minimum(abs(diff), prm.clip)
            linear_part = abs(diff) - quadratic_part
            cost = 0.5 * quadratic_part ** 2 + prm.clip * linear_part
        else:
            cost = 0.5 * diff ** 2

        cost = (cost * m_.flatten()).sum() / tensor.maximum(1., m_.sum())

        # use entropy regularization if it is using softmax.
        if prm.act_sel.lower() == 'softmax':
            cost += (cost_ent * mask_).sum() / tensor.maximum(1., mask_.sum())

        cost *= (uidx > prm.replay_start).astype('float32')  # start learning only after some updates.
        baseline_updates = []

    elif prm.learning.lower() == 'reinforce':
        if prm.mov_avg:
            R_mean = R.mean()
            R_std = R.std()
            R_mean_ = 0.9 * baseline_vars['R_mean'] + 0.1 * R_mean
            R_std_ = 0.9 * baseline_vars['R_std'] + 0.1 * R_std

            # Update baseline vars.
            baseline_updates = [(baseline_vars['R_mean'], R_mean_), 
                                (baseline_vars['R_std'], R_std_)]
        else:
            baseline_updates = []
            R_mean_ = 0.
            R_std_ = 1.

        if prm.idb:
            # input-dependent baseline
            #R_idb = tensor.dot(h[js, tensor.arange(h.shape[1]), :], tparams['R_W']) + tparams['R_b']
            h_const = theano.gradient.disconnected_grad(h)
            R_idb = tensor.nnet.sigmoid(tensor.dot(h_const.mean(0), tparams['R_W']) + tparams['R_b'])
            R_ = (R[:,None] - R_mean_ - R_idb) / tensor.maximum(1., R_std_)
        else:
            R_ = (R[:,None] - R_mean_) / tensor.maximum(1., R_std_)
        R_ = R_[:,0]
        consider_constant += [R_]

        cost_sup = (cost_p + cost_ent) * mask_
        cost_sup = cost_sup[:mixer].sum(0).mean()

        if prm.clip > 0:
            # Clipping l_prob so -log does not become too large.
            log_or_lin = (-tensor.log(l_prob + off) < prm.clip).astype('float32')
            log_or_lin = theano.gradient.disconnected_grad(log_or_lin)
            cost_pre = log_or_lin * -tensor.log(l_prob + off) + (1. - log_or_lin) * (1 - l_prob / tensor.exp(-prm.clip))
        else:
            cost_pre = -tensor.log(l_prob + off)

        cost_RL = (R_ * cost_pre + cost_ent) * mask_ 
        cost_RL = cost_RL[mixer:].sum(0).mean()

        cost = cost_sup + cost_RL

        if prm.idb:
            R0 = R[:,None] - R_mean_
            R0 = theano.gradient.disconnected_grad(R0)
            #cost += 0.01 * ((R_idb - R0) ** 2).mean()
            cost += ((R0 - R_idb) ** 2).mean()
    else:
        raise ValueError('Not a valid value for the learning parameter.' + \
                         ' Valid options are: "supervised", "reinforce", and "q-learning".')

    if prm.weight_decay > 0.:
        for name, w in tparams.items():
            #do not include bias.
            if not name.lower().startswith('b'):
                cost += prm.weight_decay * (w**2).sum()

    # replay memory.
    l_idx = l_idx.T
    dist = dist.dimshuffle((1,0,2))

    iin = [q, q_m, root_pages, acts_p, uidx, rs_q, rs_q_m, rl_idx, rt, rR]
    out = [cost, R, l_idx, pages_idx, best_doc, best_answer, mask, dist]

    if prm.learning.lower() == 'q-learning':
        scan_updates = scan_updates_a + scan_updates_b
        if prm.update_freq > 1:
            scan_updates += scan_updates_c
    else:
        scan_updates = scan_updates_a

    updates = scan_updates + baseline_updates

    return iin, out, updates, is_train, sup, max_hops, k_beam, mixer, f_pred, consider_constant


def get_root_pages(actions):
    root_pages = np.zeros((len(actions)), dtype=np.float32)
    for t, action in enumerate(actions):
        root_pages[t] = action[0]
    return root_pages


def get_acts(actions, max_hops):
    # Get correct actions (supervision signal)
    acts_p = -np.ones((max_hops+1, len(actions)), dtype=np.int32)
    for t, action in enumerate(actions):
        for kj, title_id in enumerate(action[1:]):
            acts_p[kj, t] = title_id

    return acts_p


def pred_error(f_pred, queries, actions, candidates, options, iterator, verbose=False):
    """
    Compute the error and document recall.
    f_pred: Theano function computing the prediction
    """

    n = 0.
    ns = 0.
    valid_R = 0.
    recall1 = 0.
    recall = 0. # document recall for the last page before the stop action.
    recall_all = 0. # document recall for all pages visited.
    uidx = -1
    i = 0
    for _, valid_index in iterator:

        q_i, q_m = utils.text2idx2([queries[t].lower() for t in valid_index], options['vocab'], prm.max_words_query*prm.n_consec)
        acts = [actions[t] for t in valid_index]
        cands = [candidates[t] for t in valid_index]

        #dummy acts that won't be used in the prediction
        acts_p = -np.ones((prm.max_hops_pred+1, len(q_i) * prm.k), dtype=np.int32)

        root_pages = get_root_pages([act[0] for act in acts])

        best_doc, best_answer, R, pages_idx, selected_docs, js, _, _, _ = f_pred(q_i, q_m, root_pages, acts_p, uidx)

        R_binary = np.ones_like(R)
        R_binary[R<1.0] = 0.0
        n += len(valid_index)
        valid_R += R.sum()

        all_docs = pages_idx.T.reshape((len(valid_index), (prm.max_hops_pred + 1) * prm.k))

        for j in range(len(valid_index)):

            # get correct path.
            acts_p = get_acts(acts[j], prm.max_hops_pred)

            ns += len(acts[j])
            
            # Compute the document recall.
            jc = np.minimum(np.maximum((acts_p != -1.0).astype('int32').sum(0) - 1, 0), prm.max_hops_pred)

            correct_docs = acts_p[jc, np.arange(acts_p.shape[1])]
            
            for correct_doc in correct_docs:
                # Doc recall for all pages visited
                recall_all += (correct_doc == all_docs[j]).any().astype('int32').sum()

                # doc recall for pages before stop action
                match = (correct_doc == selected_docs[j]).any()
                recall += match.astype('int32').sum()
            
                recall1 += (correct_doc == best_doc[j]).astype('int32').sum()

            if j == 0 and (i % prm.dispFreq == 0):
                print '\nQuery: ' + queries[valid_index[j]].replace('\n',' ')
                print 'Best document: ' + options['wiki'].get_article_title(best_doc[j])
                print 'Best answer: ' + utils.idx2text(best_answer[j], options['vocabinv'])

                print 'Supervised Path:',
                for page_idx in acts_p[:-1,0]:
                    if page_idx != -1:
                        print '->', options['wiki'].get_article_title(page_idx),
                print '-> Stop'

                print 'Actual Path:    ',
                for page_idx in pages_idx[:-1,0]:
                    if page_idx != -1:
                        print '->', options['wiki'].get_article_title(page_idx),
                print '-> Stop'

        i += 1
        uidx -= 1
        
    valid_R = valid_R / n
    recall1 = recall1 / n
    recall = recall / ns
    recall_all = recall_all / ns

    return valid_R, recall1, recall, recall_all


def train():

    optimizer=adam  # only adam is supported by now.
    options = locals().copy()

    print 'parameters:', str(options)
    prm_k = vars(prm).keys()
    prm_d = vars(prm)
    prm_k.sort()
    for x in prm_k:
        if not x.startswith('__'):
            print x,'=', prm_d[x]

    print 'loading dictionary...'
    vocab = utils.load_vocab(prm.vocab_path, prm.n_words)
    options['vocab'] = vocab

    options['vocabinv'] = {}
    for k,v in vocab.items():
        options['vocabinv'][v] = k

    print 'Loading Environment...'
    options['wiki'] = wiki.Wiki(prm.pages_path)
    if prm.compute_emb:
        import wiki_idx
        options['wikipre'] = wiki_idx.WikiIdx(prm.pages_idx_path)
    else:
        import wiki_emb
        options['wikipre'] = wiki_emb.WikiEmb(prm.pages_emb_path)

    print 'Loading Dataset...'
    qpp = qp.QP(prm.qp_path)
    q_train, q_valid, q_test = qpp.get_queries()
    a_train, a_valid, a_test = qpp.get_paths()
    c_train, c_valid, c_test = qpp.get_candidates() # get candidates obtained by the search engine

    if prm.aug>1:
        dic_thes = utils.load_synonyms()
        q_train = utils.augment(q_train, dic_thes)
        a_train = list(itertools.chain.from_iterable(itertools.repeat(x, prm.aug) for x in a_train))
        c_train = list(itertools.chain.from_iterable(itertools.repeat(x, prm.aug) for x in c_train))

    # This create the initial parameters as np ndarrays.
    # Dict name (string) -> np ndarray
    params, exclude_params = init_params()

    if prm.wordemb_path:
        print 'loading pre-trained word embeddings'
        params = load_wemb(params, vocab)
        options['W'] = params['W']

    if prm.reload_model:
        load_params(prm.reload_model, params)

    params_next = OrderedDict()
    if prm.learning.lower() == 'q-learning':

        if prm.update_freq > 1:
            # copy params to params_next
            for kk, kv in params.items():
                params_next[kk] = kv.copy()

        if prm.reload_mem:
            mem, mem_r = pkl.load(open(prm.reload_mem, 'rb'))
        else:
            mem = deque(maxlen=prm.replay_mem_size) # replay memory as circular buffer.
            mem_r = deque(maxlen=prm.replay_mem_size) # reward of each entry in the replay memory.

    print 'Building model'
    # This create Theano Shared Variable from the parameters.
    # Dict name (string) -> Theano Tensor Shared Variable
    # params and tparams have different copy of the weights.
    tparams = init_tparams(params)

    if prm.update_freq > 1:
        tparams_next = init_tparams(params_next)
    else:
        tparams_next = None

    baseline_vars = {}  
    if prm.learning.lower() == 'reinforce':
        if prm.mov_avg:
            R_mean = theano.shared(0.71*np.ones((1,)), name='R_mean')
            R_std = theano.shared(np.ones((1,)), name='R_std')
            baseline_vars = {'R_mean': R_mean, 'R_std': R_std}

    iin, out, updates, is_train, sup, max_hops, k_beam, mixer, f_pred, consider_constant \
            = build_model(tparams, tparams_next, baseline_vars, options)

    #get only parameters that are not in the exclude_params list
    tparams_ = OrderedDict([(kk, vv) for kk, vv in tparams.iteritems() if kk not in exclude_params])

    total_prm = 0
    learn_prm = 0
    for name, arr in params.items():
        if name not in exclude_params:
            learn_prm += arr.size
        total_prm += arr.size
    print 'Number of Parameters          :', total_prm
    print 'Number of Learnable Parameters:', learn_prm

    grads = tensor.grad(out[0], wrt=itemlist(tparams_), consider_constant=consider_constant)

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = optimizer(lr, tparams_, grads, iin, out, updates)

    print 'Optimization'

    if prm.train_size == -1:
        train_size = len(q_train)
    else:
        train_size = min(prm.train_size, len(q_train))

    if prm.valid_size == -1:
        valid_size = len(q_valid)
    else:
        valid_size = min(prm.valid_size, len(q_valid))

    if prm.test_size == -1:
        test_size = len(q_test)
    else:
        test_size = min(prm.test_size, len(q_test))


    print '%d train examples' % len(q_train)
    print '%d valid examples' % len(q_valid)
    print '%d test examples' % len(q_test)

    history_errs = []
    best_p = None

    if prm.validFreq == -1:
        validFreq = len(q_train) / prm.batch_size_train
    else:
        validFreq = prm.validFreq

    if prm.saveFreq == -1:
        saveFreq = len(q_train) / prm.batch_size_train
    else:
        saveFreq = prm.saveFreq

    uidx = 0  # the number of update done
    estop = False  # early stop
    start_time = time.time()
    

    try:
        for eidx in xrange(prm.max_epochs):
            n_samples = 0

            # Get new shuffled index for the training set.
            kf = get_minibatches_idx(len(q_train), prm.batch_size_train, shuffle=True)

            for _, train_index in kf:
                st = time.time()

                uidx += 1
                is_train.set_value(1.)
                max_hops.set_value(prm.max_hops_train) # select training dataset
                k_beam.set_value(1) # Training does not use beam search
                
                # Select the random examples for this minibatch
                queries = [q_train[t].lower() for t in train_index]
                # randomly select a path of each training example
                actions = []
                for t in train_index:
                    a = a_train[t]
                    actions.append(a[random.randint(0,len(a)-1)])
                
                if prm.learning.lower() == 'supervised':
                    sup.set_value(1.) # select supervised mode
                else:
                    sup.set_value(0.)

                # Get correct actions (supervision signal)
                acts_p =  get_acts(actions, prm.max_hops_train)

                # MIXER
                if prm.mixer > 0 and prm.learning.lower() == 'reinforce':
                    mixer.set_value(max(0, prm.max_hops_train - uidx // prm.mixer))
                else:
                    if prm.learning.lower() == 'supervised':
                        mixer.set_value(prm.max_hops_train+1)
                    else:
                        mixer.set_value(0)

                root_pages = get_root_pages(actions)                
                
                # Get the BoW for the queries.
                q_i, q_m = utils.text2idx2(queries, vocab, prm.max_words_query*prm.n_consec)
                n_samples += len(queries)
                
                if uidx > 1 and prm.learning.lower() == 'q-learning':
                    # Randomly select memories and convert them to numpy arrays.
                    idxs = np.random.choice(np.arange(len(mem)), size=len(queries))
                    rvs = []
                    for j in range(len(mem[idxs[0]])):
                        rv = []
                        for idx in idxs:
                            rv.append(mem[idx][j])

                        rvs.append(np.asarray(rv))
                else:
                    rvs = [np.zeros((len(queries),prm.max_words_query*prm.n_consec),dtype=np.float32), # rs_q
                           np.zeros((len(queries),prm.max_words_query*prm.n_consec),dtype=np.float32), # rs_q_m
                           np.zeros((len(queries),prm.max_hops_train+1),dtype=np.int32), # rl_idx
                           np.zeros((len(queries),prm.max_hops_train+1),dtype=np.float32), # rt
                           np.zeros((len(queries),prm.max_hops_train+1),dtype=np.float32) # rr
                          ]


                cost, R, l_idx, pages_idx, best_doc, best_answer, mask, dist \
                        = f_grad_shared(q_i, q_m, root_pages, acts_p, uidx, *rvs)
                f_update(prm.lrate)

                if prm.learning.lower() == 'q-learning': 
                    # update weights of the next_q_val network.
                    if prm.update_freq > 1 and ((uidx % prm.update_freq == 0) or (uidx == prm.replay_start)):
                        for tk, tv in tparams.items():
                            if tk in tparams_next:
                                tparams_next[tk].set_value(tv.get_value().copy())

                    # Only update memory after freeze_mem or before replay_start.
                    if uidx < prm.replay_start or uidx > prm.freeze_mem:
                        # Update Replay Memory.
                        t = np.zeros((len(queries), prm.max_hops_train+1))
                        rR = np.zeros((len(queries), prm.max_hops_train+1))
                        pr = float(np.asarray(mem_r).sum()) / max(1., float(len(mem_r)))

                        for i in range(len(queries)):
                            j = np.minimum(mask[:,i].sum(), prm.max_hops_train)
                            # If the agent chooses to stop or the episode ends,
                            # the reward will be the reward obtained with the chosen document.
                            rR[i,j] = R[i]
                            t[i,j] = 1.
                            
                            add = True
                            if prm.prioritized_sweeping >= 0 and uidx > 1:
                                # Prioritized_sweeping: keep the percentage of memories
                                # with reward=1 approximately equal to <prioritized_sweeping>.
                                if ((pr < prm.prioritized_sweeping - 0.05) and (rR[i,j] == 0.)) or ((pr > prm.prioritized_sweeping + 0.05) and (rR[i,j] == 1.)):
                                    add = False

                            if add:
                                mem.append([q_i[i], q_m[i], l_idx[i], t[i], rR[i]])
                                mem_r.append(rR[i])

                if np.isnan(cost) or np.isinf(cost):
                    print 'NaN detected'
                    return 1., 1., 1.
    
                #if uidx % 100 == 0:
                #    vis_att(pages_idx[:,-1], queries[-1], alpha[:,-1,:], uidx, options)

                if np.mod(uidx, prm.dispFreq) == 0:
                    print "\nQuery: " + queries[0].replace("\n"," ")
       
                    print 'Supervised Path:',
                    for i, page_idx in enumerate(acts_p[:-1,0]):
                        if page_idx != -1:
                            print '->', options['wiki'].get_article_title(page_idx),
                    print '-> Stop'

                    print 'Actual Path:    ',
                    for i, page_idx in enumerate(pages_idx[:-1,0]):
                        if page_idx != -1:
                            print '->', options['wiki'].get_article_title(page_idx),
                    print '-> Stop'

                    print 'Best Document: ' + options['wiki'].get_article_title(best_doc[0])
                    print 'Best Answer: ' + utils.idx2text(best_answer[0], options['vocabinv'])
                    print('Epoch '+ str(eidx) + ' Update '+ str(uidx) + ' Cost ' + str(cost) + \
                               ' Reward Mean ' + str(R.mean()) + ' Reward Max ' + str(R.max()) + \
                               ' Reward Min ' + str(R.min()) + \
                               ' Q-Value Max (avg per sample) ' + str(dist.max(2).mean()) + \
                               ' Q-Value Mean ' + str(dist.mean()))

                    if prm.learning.lower() == 'q-learning':
                        pr = float(np.asarray(mem_r).sum()) / max(1., float(len(mem_r)))
                        print 'memory replay size:', len(mem), ' positive reward:', pr

                    print 'Time per Minibatch Update: ' + str(time.time() - st)
                       


                if np.mod(uidx, validFreq) == 0 or uidx == 1:
             
                    kf_train = get_minibatches_idx(len(q_train), prm.batch_size_pred, shuffle=True, max_samples=train_size)
                    kf_valid = get_minibatches_idx(len(q_valid), prm.batch_size_pred, shuffle=True, max_samples=valid_size)
                    kf_test = get_minibatches_idx(len(q_test), prm.batch_size_pred, shuffle=True, max_samples=test_size)

                    is_train.set_value(0.)
                    sup.set_value(0.) # supervised mode off
                    mixer.set_value(0) # no supervision
                    max_hops.set_value(prm.max_hops_pred)
                    k_beam.set_value(prm.k)

                    print '\nEvaluating Training Set'
                    train_R, train_recall1, train_recall, train_recall_all, \
                        = pred_error(f_pred, q_train, a_train, c_train, options, kf_train)

                    print '\nEvaluating Validation Set'
                    valid_R, valid_recall1, valid_recall, valid_recall_all, \
                         = pred_error(f_pred, q_valid, a_valid, c_valid, options, kf_valid)

                    print '\nEvaluating Test Set'
                    test_R, test_recall1, test_recall, test_recall_all, \
                        = pred_error(f_pred, q_test, a_test, c_test, options, kf_test)

                    history_errs.append([valid_recall, test_recall])

                    if (uidx == 0 or
                        valid_recall >= np.array(history_errs)[:,0].min()):

                        best_p = unzip(tparams)
                        bad_counter = 0

                    print 'Reward Train', train_R, ' Valid', valid_R, ' Test', test_R
                    print 'Recall@1 Train ' + str(train_recall1), ' Valid', valid_recall1, ' Test',test_recall1
                    print 'Recall@' + str(prm.k), ' Train', train_recall, ' Valid', valid_recall, ' Test',test_recall
                    print 'Recall@' + str(prm.max_hops_pred * prm.k), ' Train', train_recall_all, ' Valid', valid_recall_all, ' Test', test_recall_all

                    if (len(history_errs) > prm.patience and
                        valid_recall <= np.array(history_errs)[:-prm.patience,
                                                               0].min()):
                        bad_counter += 1
                        if bad_counter > prm.patience:
                            print 'Early Stop!'
                            estop = True
                            break


                if prm.saveto and np.mod(uidx, saveFreq) == 0:
                    print 'Saving...',

                    if best_p is not None:
                        params = best_p
                    else:
                        params = unzip(tparams)
                    np.savez(prm.saveto, history_errs=history_errs, **params)
                    #pkl.dump(options, open('%s.pkl' % prm.saveto, 'wb'), -1)
                    print 'Done'

                if prm.learning.lower() == 'q-learning':
                    if prm.saveto_mem and np.mod(uidx, saveFreq) == 0:
                        pkl.dump([mem, mem_r], open(prm.saveto_mem, 'wb'), -1)


            print 'Seen %d samples' % n_samples

            if estop:
                break

    except KeyboardInterrupt:
        print "Training interupted"

    return


if __name__ == '__main__':
    # See parameters.py for all possible parameter and their definitions.
    train()
