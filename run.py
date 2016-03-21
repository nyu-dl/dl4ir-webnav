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
import wiki_emb
import qp
import parameters as prm
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab since the server might not have an X server.
import matplotlib.pyplot as plt
from nltk.tokenize import wordpunct_tokenize
import copy

# compute_test_value is 'off' by default, meaning this feature is inactive
#theano.config.compute_test_value = 'warn' # Use 'warn' to activate this feature

# Set the random number generators' seeds for consistency
SEED = 123
np.random.seed(SEED)


def vis_att(pages_idx, query, data, it, options):

    rows = [prm.root_page]
    for pageidx in pages_idx[:-1]:
        rows.append(options['wiki'].get_article_title(pageidx).decode('utf-8', 'ignore'))

    columns = []
    for word in wordpunct_tokenize(query.lower()):
        if word in options['vocab']:
            columns.append(word.decode('utf-8', 'ignore'))
    columns = columns[:prm.max_words_query*prm.n_consec]

    data = data[:,:len(columns)]

    fig,ax=plt.subplots(figsize=(24,8))
    #Advance color controls
    im = ax.pcolor(data,cmap=plt.cm.gray,edgecolors='w')
    fig.colorbar(im)
    ax.set_xticks(np.arange(0,len(columns))+0.5)
    ax.set_yticks(np.arange(0,len(rows))+0.5)
    ax.tick_params(axis='x', which='minor', pad=15)
    # Here we position the tick labels for x and y axis
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    plt.xticks(rotation=90)
    fig.subplots_adjust(bottom=0.2)
    fig.subplots_adjust(left=0.2)
    #Values against each labels
    ax.set_xticklabels(columns,minor=False,fontsize=10)
    ax.set_yticklabels(rows,minor=False,fontsize=14)
    plt.savefig('vis_'+str(it)+'.png')
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
                                        p=0.5, n=1,
                                        dtype=state_before.dtype)),
                         state_before * 0.5)
    return proj


def load_params(path, params):
    pp = np.load(path)
    for kk, vv in params.iteritems():
        if kk in pp:
            params[kk] = pp[kk]
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
    return e_x / (e_x.sum(axis=-1, keepdims=True) + 1e-8) #this small constant avoids possible division by zero created by the mask


def init_params():
    params = OrderedDict()

    if prm.encoder.lower() == 'lstm':
        mul = 4
    else:
        mul = 1

    for i in range(prm.n_layers):

        # compatibility wiht older versions.
        if i == 0: 
            i = ''
        else:
            i = str(i)

        params['E_L'+i] = 0.01 * np.random.randn(prm.dim_emb, mul * prm.dim_proj).astype(config.floatX) # section to hidden state, link step
        params['E_Q'+i] = 0.01 * np.random.randn(prm.dim_emb, mul * prm.dim_proj).astype(config.floatX) # question to hidden state, link step
        params['U_I'+i] = 0.01 * np.random.randn(prm.dim_proj, mul * prm.dim_proj).astype(config.floatX) # transition link
        params['b'+i] = 0.1 * np.ones((mul * prm.dim_proj,)).astype(config.floatX) # bias

    params['stop'] = 0.01 * np.random.randn(prm.dim_emb).astype(config.floatX) # stop action vector
    params['U_O'] = 0.01 * np.random.randn(prm.dim_proj, prm.dim_emb).astype(config.floatX) # score
    params['b_U_O'] = 0.1 * np.ones((prm.dim_proj,)).astype(config.floatX) # bias
    params['U_R'] = 0.01 * np.random.randn(prm.dim_proj, 1).astype(config.floatX) # score    
    params['b_U_R'] = np.zeros((1,)).astype(config.floatX) # bias

    if prm.att_query:
        params['Ww_att_q'] = 0.01 * np.random.randn(prm.dim_emb, prm.att_window).astype(config.floatX) # query
        params['Wq_att_q'] = 0.01 * np.random.randn(prm.dim_emb, prm.dim_emb).astype(config.floatX) # query
        params['Wh_att_q'] = 0.01 * np.random.randn(prm.dim_proj, prm.dim_emb).astype(config.floatX) # hidden state
        params['Wl_att_q'] = 0.01 * np.random.randn(prm.dim_emb, prm.dim_emb).astype(config.floatX) # link embedding
        params['bq_att_q'] = 0.1 * np.ones((prm.dim_emb,)).astype(config.floatX) # bias
        params['We_att_q'] = 0.01 * np.random.randn(prm.dim_proj, 1).astype(config.floatX) # score
        params['bw_att_q'] = 0.1 * np.ones((prm.dim_emb,)).astype(config.floatX) # bias score
        params['be_att_q'] = 0.1 * np.ones((1,)).astype(config.floatX) # bias score

    if prm.att_doc:
        params['Ww_att_d'] = 0.01 * np.random.randn(prm.dim_emb, prm.att_window).astype(config.floatX) # query
        params['Wq_att_d'] = 0.01 * np.random.randn(prm.dim_emb, prm.dim_emb).astype(config.floatX) # query
        params['Wh_att_d'] = 0.01 * np.random.randn(prm.dim_proj, prm.dim_emb).astype(config.floatX) # hidden state
        params['Wl_att_d'] = 0.01 * np.random.randn(prm.dim_emb, prm.dim_emb).astype(config.floatX) # link embedding
        params['bq_att_d'] = 0.1 * np.ones((prm.dim_emb,)).astype(config.floatX) # bias
        params['We_att_d'] = 0.01 * np.random.randn(prm.dim_proj, 1).astype(config.floatX) # score
        params['bw_att_d'] = 0.1 * np.ones((prm.dim_emb,)).astype(config.floatX) # bias score
        params['be_att_d'] = 0.1 * np.ones((1,)).astype(config.floatX) # bias score

    if prm.idb:
        params['R_W'] = 0.01 * np.random.randn(prm.dim_proj, 1).astype(config.floatX) # question   
        params['R_b'] = 0.1 * np.ones((1,)).astype(config.floatX) # bias

    params['W'] = 0.01 * np.random.randn(prm.n_words, prm.dim_emb).astype(config.floatX) # vocab to word embeddings

    exclude_params = {}
    if prm.fixed_wemb:
        exclude_params['W'] = True
        
    return params, exclude_params


def lstm_layer(q_at, l_a_, h_, c_, m_, tparams, layer_num=''):
    layer_num = str(layer_num)
    
    preact = tensor.dot(q_at, tparams['E_Q'+layer_num])
    preact += tensor.dot(h_, tparams['U_I'+layer_num])
    preact += tensor.dot(l_a_, tparams['E_L'+layer_num])
    preact += tparams['b'+layer_num]
    
    if prm.encoder.lower() == 'lstm':
        i = tensor.nnet.sigmoid(_slice(preact, 0, prm.dim_proj))
        f = tensor.nnet.sigmoid(_slice(preact, 1, prm.dim_proj))
        o = tensor.nnet.sigmoid(_slice(preact, 2, prm.dim_proj))
        c = tensor.tanh(_slice(preact, 3, prm.dim_proj))
    
        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_
    
        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_
    else:
        c = c_
        h = tensor.tanh(preact) * m_[:, None]
    return h, c


def val(q_a, q_m, h_, l_a_, c_, m_, L_a, L_m, tparams, k_beam, n_samples, uidx):
    
    n_links = L_a.shape[1] + 1
    
    if prm.att_query:
        pad = tensor.alloc(np.array(0., dtype=np.float32), q_a.shape[0], (prm.att_window-1)/2, prm.dim_emb)
        q_aw_ = tensor.concatenate([pad, q_a, pad], axis=1)
        q_aw = q_aw_[:,:-(prm.att_window-1),:] * tparams['Ww_att_q'][:,0]
        # convolution embeddings
        for i in range(1,prm.att_window):
            q_aw += q_aw_[:,i:q_aw_.shape[1]-(prm.att_window-1)+i,:] * tparams['Ww_att_q'][:,i]
        q_aw += tparams['bw_att_q']

        e = tensor.dot(q_aw, tparams['Wq_att_q'])
        e += tensor.dot(h_, tparams['Wh_att_q'])[:,None,:]
        e += tensor.dot(l_a_, tparams['Wl_att_q'])[:,None,:]
        e += tparams['bq_att_q']
        e = tensor.tanh(e)
        e = tensor.dot(e, tparams['We_att_q']) + tparams['be_att_q']
        e = e.reshape((e.shape[0],e.shape[1]))

        # repeat for beam search
        q_m_ = tensor.extra_ops.repeat(q_m, k_beam, axis=0)
        alpha = softmax_mask(e, q_m_)
        q_at = (alpha[:,:,None] * q_a).sum(1)
    else:
        alpha = tensor.alloc(np.array(0., dtype=np.float32), q_a.shape[0], q_a.shape[1])
        q_at = q_a


    h, c = lstm_layer(q_at, l_a_, h_, c_, m_, tparams)
    # Multi-layer lstm
    for i in range(1,prm.n_layers):
        h, c = lstm_layer(q_at, l_a_, h, c, m_, tparams, layer_num=i)

    if prm.att_doc:
        pad = tensor.alloc(np.array(0., dtype=np.float32), L_a.shape[0], L_a.shape[1], (prm.att_window-1)/2, prm.dim_emb)
        L_aw_ = tensor.concatenate([pad, L_a, pad], axis=2)
        L_aw = L_aw_[:,:,:-(prm.att_window-1),:] * tparams['Ww_att_d'][:,0]
        # Convolution embeddings
        for i in range(1,prm.att_window):
            L_aw += L_aw_[:,:,i:L_aw_.shape[2]-(prm.att_window-1)+i,:] * tparams['Ww_att_d'][:,i]
        L_aw += tparams['bw_att_d']

        e = tensor.dot(L_aw, tparams['Wl_att_d'])
        e += tensor.dot(h, tparams['Wh_att_d'])[:,None,None,:]
        e += tensor.dot(q_at, tparams['Wq_att_d'])[:,None,None,:]
        e += tparams['bq_att_d']
        e = tensor.tanh(e)
        e = tensor.dot(e, tparams['We_att_d']) + tparams['be_att_d']
        e = e.reshape((e.shape[0],e.shape[1],e.shape[2]))

        # Repeat for beam search
        alpha = softmax_mask(e, L_m)
        L_at = (alpha[:,:,:,None] * L_a).sum(2)
        L_m = (L_m.sum(2) > 0.).astype('float32') 
    else:
        L_at = L_a

    # Append stop action
    stop = tparams['stop'][None, None, :]
    stop = tensor.extra_ops.repeat(x=stop, repeats=n_samples * k_beam, axis=0)
    L_as = tensor.concatenate([L_at, stop], axis=1)
    stop_m = tensor.alloc(np_floatX(1.), n_samples * k_beam, 1)
    L_ms = tensor.concatenate([L_m, stop_m], axis=1)

    z = tensor.tanh(tensor.dot(h, tparams['U_O']) + tparams['b_U_O'])
    
    res = tensor.dot(L_as* z[:,None,:], tparams['U_R']) + tparams['b_U_R']
    res = res.reshape((n_samples, k_beam * n_links)) # Reshape for beam search
    L_ms = L_ms.reshape((n_samples, k_beam * n_links))

    score = res * L_ms

    return score, h, c, L_as, L_ms


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


def ff(q, q_m, k_beam, trng, is_train, options, uidx, tparams, mixer, sup, root_pages, max_hops, acts_p, rl_idx=None, get_links=None):

    n_samples = q.shape[0]
    off = 1e-8
    epsilon = tensor.maximum(prm.epsilon_min, prm.epsilon_start - (prm.epsilon_start - prm.epsilon_min) * (uidx / prm.epsilon_decay))

    if not get_links:
        get_links = Link(options['wiki'], options['wikiemb'], options['vocab']) # instantiate custom function to get wiki links

    def _step(it, act_p, p_, m_, l_a_, h_, c_, q_a):

        L_a, L_m, l_page, l_truth = get_links(p_, act_p, it, uidx)
        
        dist, h, c, L_as, L_ms = val(q_a, q_m, h_, l_a_, c_, m_, L_a, L_m, tparams, k_beam, n_samples, uidx)

        n_links = L_as.shape[1]

        # Select hyperlink by sampling from the distribution.
        if prm.learning.lower() == 'q_learning':

            if rl_idx: #if this is the replay memory pass, just use the q-value function
                dist = tensor.nnet.sigmoid(dist) * L_ms
                res_ = dist.argmax(1)

            else: # otherwise, select actions using epsilon-greedy or softmax.

                if prm.act_sel.lower() == 'epsilon-greedy':
                    dist = tensor.nnet.sigmoid(dist) * L_ms

                    greedy = tensor.eq(is_train,1.).astype('float32') * (trng.uniform(size=(n_samples,)) > epsilon) + \
                             (1. - tensor.eq(is_train,1.).astype('float32'))

                    randd = tensor.floor(trng.uniform(size=(n_samples,)) * L_ms.sum(1)).astype('int32') - 1

                    # Convert the stop action (idx = -1) to the last index.
                    randd = tensor.eq(randd, -1).astype('float32') * (L_ms.shape[1] - 1) + tensor.neq(randd, -1).astype('float32') * randd

                    res_pre = tensor.eq(it, 0.).astype('int32') * dist[:,:n_links].argsort(axis=1)[:, -k_beam:].flatten().astype("int32") \
                           + (1 - tensor.eq(it, 0.).astype('int32')) * dist.argsort(axis=1)[:, -k_beam:].reshape((n_samples * k_beam,)).astype("int32")

                    # Repeat for beam search
                    greedy = tensor.extra_ops.repeat(greedy, k_beam, axis=0)
                    randd = tensor.extra_ops.repeat(randd, k_beam, axis=0)

                    res_ = (1. - greedy) * randd + greedy * res_pre
                    #res_ = res_pre

                elif prm.act_sel.lower() == 'softmax':
                    dist = softmax_mask(dist, L_ms)

                    lp_ = tensor.eq(is_train,1.).astype('float32') * trng.multinomial(n=1, pvals=dist, dtype=dist.dtype) \
                           + (1. - tensor.eq(is_train,1.).astype('float32')) * dist

                    res_ = tensor.eq(it, 0.).astype('int32') * lp_[:,:n_links].argsort(axis=1)[:, -k_beam:].flatten().astype("int32") \
                           + (1 - tensor.eq(it, 0.).astype('int32')) * lp_.argsort(axis=1)[:, -k_beam:].reshape((n_samples * k_beam,)).astype("int32")

        else:
            dist = softmax_mask(dist, L_ms)

            lp_ = tensor.eq(is_train,1.).astype('float32') * trng.multinomial(n=1, pvals=dist, dtype=dist.dtype) \
                      + (1. - tensor.eq(is_train,1.).astype('float32')) * dist

            res_ = tensor.eq(it, 0.).astype('int32') * lp_[:,:n_links].argsort(axis=1)[:, -k_beam:].flatten().astype("int32") \
                   + (1 - tensor.eq(it, 0.).astype('int32')) * lp_.argsort(axis=1)[:, -k_beam:].reshape((n_samples * k_beam,)).astype("int32")

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

        # supervised only: compute the cost for page selection
        cost_p = -tensor.log(dist[tensor.arange(dist.shape[0]), l_truth] + off)

        # check if the stop action was chosen.
        m = tensor.neq(l_idx1, n_links-1).astype("float32")

        # consider previous mask.
        m *= m_ 
        
        # Get indices of the next articles.
        p = l_page[l_idx0, l_idx1]

        return p, m, l_a, h, c, l_prob, l_ent, cost_p, l_idx, dist

    #get embeddings for the queries
    q_a = tparams['W'][q.flatten()].reshape((q.shape[0], q.shape[1], prm.dim_emb)) * q_m[:,:,None]

    if not prm.att_query:
        q_a = q_a.sum(1) / tensor.maximum(1., q_m.sum(1, keepdims=True))

    #repeat question for beam search
    q_a = tensor.extra_ops.repeat(q_a, k_beam, axis=0)
    root_pages_ = tensor.extra_ops.repeat(root_pages, k_beam)

    (pages_idx, mask, _, h, _, l_prob, l_ent, cost_p, l_idx, dist), scan_updates = theano.scan(_step,
                                sequences=[tensor.arange(max_hops+1), acts_p],
                                outputs_info=[root_pages_, #page idx
                                              tensor.alloc(np.array(1., dtype=np.float32), k_beam * n_samples),    # mask
                                              tensor.alloc(np_floatX(0.),  # l_a
                                                           k_beam * n_samples,
                                                           prm.dim_emb),
                                              tensor.alloc(np_floatX(0.),  # h
                                                           k_beam * n_samples,
                                                           prm.dim_proj),
                                              tensor.alloc(np_floatX(0.),  # c
                                                           k_beam * n_samples,
                                                           prm.dim_proj),
                                              None,  # l_prob
                                              None,  # l_ent
                                              None,  # cost_p
                                              None,  # l_idx
                                              None,  # dist
                                              ],
                                non_sequences=[q_a],
                                name='lstm_layers',
                                n_steps=max_hops+1)

    return (pages_idx, mask,  h, l_prob, l_ent, cost_p, root_pages_, l_idx, dist), scan_updates, get_links


def build_model(tparams, tparams_next, baseline_vars, options):
    trng = RandomStreams(SEED)
    off = 1e-8  # small constant to avoid log 0 = -inf
    consider_constant = []

    is_train = theano.shared(np_floatX(0.)) # Used for dropout.
    mixer = theano.shared(np.asarray(0, dtype=np.int32)) # Used for MIXER.
    sup = theano.shared(np_floatX(0.)) # Supervised or not
    max_hops = theano.shared(np.asarray(0, dtype=np.int32)) # Max number of iterations
    k_beam = theano.shared(np.asarray(0, dtype=np.int32)) # top-k items in the beam search.
    
    q = tensor.imatrix('q')
    q_m = tensor.fmatrix('q_m')
    root_pages = tensor.fvector('root_pages')
    acts_p = tensor.fmatrix('acts_p')

    #used only when prm.learning = 'q_learning'
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
    acts_p.tag.test_value = np.zeros((prm.max_hops_train+1,prm.batch_size_train), dtype=theano.config.floatX)
    uidx.tag.test_value = np.zeros((1,), dtype='int32')
    rs_q_a.tag.test_value = np.zeros((prm.batch_size_train,prm.dim_emb), dtype=theano.config.floatX)
    rs_q_m.tag.test_value = np.zeros((prm.batch_size_train,prm.n_consec*prm.max_words_query), dtype=theano.config.floatX)
    rl_idx.tag.test_value = np.zeros((prm.batch_size_train,), dtype='int32')
    rt.tag.test_value = np.zeros((prm.batch_size_train,), dtype=theano.config.floatX)
    rR.tag.test_value = np.zeros((prm.batch_size_train,), dtype=theano.config.floatX)
    """
    
    n_samples = q.shape[0]
    
    (pages_idx, mask, h, l_prob, l_ent, cost_p, root_pages_, l_idx, dist), scan_updates_a, _ = \
        ff(q, q_m, k_beam, trng, is_train, options, uidx, tparams, mixer, sup, root_pages, max_hops, acts_p)

    # Get only the used probabilities.
    mask_ = tensor.concatenate([tensor.alloc(np_floatX(1.), 1, mask.shape[1]), mask], axis=0)[:-1,:]
    l_prob *= mask_   # l_prob.shape = (n_iterations, n_samples)
    l_ent *= mask_   # l_ent.shape = (n_iterations, n_samples)

    get_sent = Sentence(options['wiki'], options['vocab'], prm.n_consec) # instantiate custom function to get sentences 
    pages_idx_ = tensor.concatenate([root_pages_[None,:], pages_idx[:-1]], axis=0)

    # get last valid action before the stop action. In case the all the mask is True, get the last action.
    j = (tensor.minimum(tensor.alloc(mask.shape[0] - 1, mask.shape[1]), mask.sum(axis=0))).astype("int32") 
    pps = pages_idx_.T
    pps = pps[tensor.arange(pps.shape[0]),j]

    R, best_page_idx, best_answer = get_sent(q, q_m, pps, k_beam)
    
    f_pred = theano.function([q, q_m, root_pages, acts_p, uidx], \
                             [best_answer, best_page_idx, R, pages_idx.reshape((max_hops+1,n_samples,k_beam))], \
                              updates=scan_updates_a, name='f_pred', on_unused_input='ignore')

    # entropy regularization
    cost_ent = -prm.erate * l_ent
    
    if prm.learning.lower() == 'supervised':
        cost = ((cost_p + cost_ent) * mask_).sum(0).mean()
        baseline_updates = []

    elif prm.learning.lower() == 'q_learning':

        (_, m, _, _, _, _, _, _, q_vals), scan_updates_b, get_links = \
                ff(rs_q, rs_q_m, k_beam, trng, is_train, \
                    options, uidx, tparams, mixer, sup, \
                    root_pages, max_hops, acts_p, rl_idx)

        m = m.T
        m_ = tensor.concatenate([tensor.alloc(np_floatX(1.), m.shape[0], 1), m], axis=1)[:,:-1]

        q_vals = q_vals.dimshuffle((1,0,2))
        
        if prm.update_freq > 0:
            (_, _, _, _, _, _, _, _, n_q_vals), scan_updates_c, _ = \
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


        q_vals_ = q_vals.reshape((-1,prm.max_links+1))
        n_q_vals_ = n_q_vals.reshape((-1,prm.max_links+1))
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
        R_mean = R.mean()
        R_std = R.std()
        R_mean_ = 0.9 * baseline_vars['R_mean'] + 0.1 * R_mean
        R_std_ = 0.9 * baseline_vars['R_std'] + 0.1 * R_std

        # Update baseline vars.
        baseline_updates = [(baseline_vars['R_mean'], R_mean_), 
                            (baseline_vars['R_std'], R_std_)]

        if prm.idb:
            # input-dependent baseline
            #R_idb = tensor.dot(h[j, tensor.arange(h.shape[1]), :], tparams['R_W']) + tparams['R_b']
            R_idb = tensor.dot(h.mean(0), tparams['R_W']) + tparams['R_b']
            R_ = (R[:,None] - R_mean_ - R_idb) / tensor.maximum(1., R_std_)
        else:
            R_ = (R[:,None] - R_mean_) / tensor.maximum(1., R_std_)
        R_ = R_[:,0]
        consider_constant += [R_]

        cost_sup = (cost_p + cost_ent) * mask_
        cost_sup = cost_sup[:mixer].sum(0).mean()

        if prm.clip > 0:
            # Clipping l_prob so -log does not become too large.
            log_or_lin = (-tensor.log(l_prob) < prm.clip).astype('float32')
            cost_pre = log_or_lin * -tensor.log(l_prob + off) + (1. - log_or_lin) * (1 - l_prob / tensor.exp(-prm.clip))
        else:
            cost_pre = -tensor.log(l_prob + off)

        cost_RL = (R_ * cost_pre + cost_ent) * mask_ 
        cost_RL = cost_RL[mixer:].sum(0).mean()

        cost = cost_sup + cost_RL

        if prm.idb:
            R0 = R[:,None] - R_mean_
            consider_constant += [R0]
            cost += 0.01 * ((R_idb - R0) ** 2).mean()
    else:
        raise ValueError('Not a valid value for the learning parameter.' + \
                         ' Valid options are: "supervised", "reinforce", and "q_learning".')

    # Experience replay memory.
    l_idx = l_idx.T
    mask = mask.T
    dist = dist.dimshuffle((1,0,2))

    iin = [q, q_m, root_pages, acts_p, uidx, rs_q, rs_q_m, rl_idx, rt, rR]
    out = [cost, R, l_idx, pages_idx, best_page_idx, best_answer, mask, dist]

    if prm.learning.lower() == 'q_learning':
        scan_updates = scan_updates_a + scan_updates_b
        if prm.update_freq > 0:
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


def get_acts(actions, max_hops, k_beam):
    # Get correct actions (supervision signal)
    acts_p = -np.ones((max_hops+1, len(actions)), dtype=np.float32)
    for t, action in enumerate(actions):
        for kj, title_id in enumerate(action[1:]):
            acts_p[kj, t] = title_id

    # repeat for the beam search
    acts_p = np.repeat(acts_p, k_beam, axis=1)
    return acts_p


def pred_error(f_pred, queries, actions, options, iterator, verbose=False):
    """
    Just compute the error
    f_pred: Theano functin computing the prediction
    """

    valid_acc = np.zeros((prm.max_hops_train + 2), dtype=np.float32)
    valid_R = np.zeros((prm.max_hops_train + 2), dtype=np.float32)
    n = np.zeros((prm.max_hops_train + 2), dtype=np.float32)
    acts_pc = 0.
    acts_pt = 0.
    uidx = -1
    visited_pages = []            

    for _, valid_index in iterator:
        q_i, q_m = utils.text2idx2([queries[t].lower() for t in valid_index], options['vocab'], prm.max_words_query*prm.n_consec)
        acts = [actions[t] for t in valid_index]

        #fake acts that won't be used in the prediction
        acts_p = -np.ones((prm.max_hops_pred+1, len(q_i) * prm.k), dtype=np.float32)
        
        root_pages = get_root_pages(acts)

        best_answer, best_page_idx, R, pages_idx = f_pred(q_i, q_m, root_pages, acts_p, uidx)

        pages_idx_ = np.swapaxes(pages_idx,0,1)
        pages_idx_ = pages_idx_.reshape(pages_idx_.shape[0],-1)

        #get pages visited:
        for page_idx in pages_idx_:
            visited_pages.append([])
            for idx in page_idx:
                if idx != -1:
                    visited_pages[-1].append(idx)

        R_binary = np.ones_like(R)
        R_binary[R<1.0] = 0.0
        n[-1] += len(valid_index)
        valid_R[-1] += R.sum()
        valid_acc[-1] += R_binary.sum()
        
        # get correct page-actions.
        acts_p = get_acts(acts, prm.max_hops_pred, prm.k)

        pages_idx = pages_idx.reshape((pages_idx.shape[0],-1))

        # Check how many page actions the model got right.
        mask_pc = np.logical_or((pages_idx != -1.0), (acts_p != -1.0)).astype('float32')
        acts_pc += ((pages_idx == acts_p).astype('float32') * mask_pc).sum()
        acts_pt += mask_pc.sum() #total number of actions

        # compute accuracy per hop
        for i in range(prm.max_hops_train+1):
            n_hops = (acts_p != -1.0).astype('float32').sum(0)
            n_hops= n_hops.reshape((-1, prm.k))[:,0] # beam search use only the first n_samples actions
            ih = (n_hops==i)
            valid_R[i] += R[ih].sum()
            valid_acc[i] += R_binary[ih].sum()
            n[i] += ih.astype('float32').sum()

        with open(prm.outpath, 'a') as fout:
            fout.write("\n\nQuery: " + queries[valid_index[-1]].replace("\n"," "))
            nh = (acts_p[:,-1] != -1.0).astype('int32').sum()
            if nh == 0:
                fout.write('\nCorrect Path: ' + options['wiki'].get_article_title(int(root_pages[-1])))
            else:
                path = ''
                for a in acts_p[:nh, -1]:
                    path += ' -> ' + options['wiki'].get_article_title(int(a))
                fout.write('\nCorrect Path: ' + path)

            fout.write('\nNumber of hops: ' + str(int(nh)))
            fout.write('\nBest answer: ' + utils.idx2text(best_answer[-1], options['vocabinv']))
            fout.write('\nBest page: ' + options['wiki'].get_article_title(best_page_idx[-1]))
            for i, pageidx in enumerate(pages_idx[:,-1]):
                fout.write('\niteration: ' +str(i) + " page idx " + str(pageidx) + ' title '+ options['wiki'].get_article_title(pageidx))

        uidx -= 1
        
    valid_R = valid_R / n
    valid_err = 1 - valid_acc / n
    acts_pc = acts_pc / acts_pt

    return valid_err, valid_R, acts_pc, visited_pages


def train_lstm():

    optimizer=adam  # only adam is supported by now.
    options = locals().copy()
    with open(prm.outpath, "a") as fout:
        fout.write("parameters:" + str(options) + str(prm.__dict__))

    print "loading dictionary..."
    vocab = utils.load_vocab(prm.vocab_path, prm.n_words)
    options['vocab'] = vocab

    options['vocabinv'] = {}
    for k,v in vocab.items():
        options['vocabinv'][v] = k

    print 'Loading data...'
    options['wiki'] = wiki.Wiki(prm.pages_path)
    options['wikiemb'] = wiki_emb.WikiEmb(prm.pages_emb_path)

    #load Q&A Wiki dataset
    qpp = qp.QP(prm.qp_path)
    q_train, q_valid, q_test = qpp.get_queries()
    a_train, a_valid, a_test = qpp.get_paths()

    print 'Building model'
    # This create the initial parameters as np ndarrays.
    # Dict name (string) -> np ndarray
    params, exclude_params = init_params()

    if prm.wordemb_path:
        print 'loading pre-trained weights for word embeddings'
        params = load_wemb(params, vocab)
        options['W'] = params['W']

    if prm.reload_model:
        load_params(prm.reload_model, params)

    params_next = OrderedDict()
    if prm.learning.lower() == 'q_learning' and prm.update_freq > 0:
        # copy params to params_next
        for kk, kv in params.items():
            params_next[kk] = kv.copy()

    # This create Theano Shared Variable from the parameters.
    # Dict name (string) -> Theano Tensor Shared Variable
    # params and tparams have different copy of the weights.
    tparams = init_tparams(params)

    if prm.update_freq > 0:
        tparams_next = init_tparams(params_next)
    else:
        tparams_next = None
  
    if prm.learning.lower() == 'reinforce':
        R_mean = theano.shared(0.71*np.ones((1,)), name='R_mean')
        R_std = theano.shared(np.ones((1,)), name='R_std')
        baseline_vars = {'R_mean': R_mean, 'R_std': R_std}
    else:
        baseline_vars = {}

    iin, out, updates, is_train, sup, max_hops, k_beam, mixer, f_pred, consider_constant \
            = build_model(tparams, tparams_next, baseline_vars, options)

    #get only parameters that are not in the exclude_params list
    tparams_ = OrderedDict([(kk, vv) for kk, vv in tparams.iteritems() if kk not in exclude_params])

    grads = tensor.grad(out[0], wrt=itemlist(tparams_), consider_constant=consider_constant)

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = optimizer(lr, tparams_, grads, iin, out, updates)

    print 'Optimization'

    if prm.train_size == -1:
        train_size = len(q_train)
    else:
        train_size = prm.train_size

    if prm.valid_size == -1:
        valid_size = len(q_valid)
    else:
        valid_size = prm.valid_size

    if prm.test_size == -1:
        test_size = len(q_test)
    else:
        test_size = prm.test_size

    with open(prm.outpath, "a") as fout:
        fout.write("\n%d train examples" % len(q_train)) 
    with open(prm.outpath, "a") as fout:
        fout.write("\n%d valid examples" % len(q_valid)) 
    with open(prm.outpath, "a") as fout:
        fout.write("\n%d test examples" % len(q_test))

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
    
    experience = deque(maxlen=prm.replay_mem_size) # experience replay memory as circular buffer.
    experience_r = deque(maxlen=prm.replay_mem_size) # reward of each entry in the replay memory.

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
                actions = [a_train[t] for t in train_index]
                
                if prm.learning.lower() == 'supervised':
                    sup.set_value(1.) # select supervised mode
                else:
                    sup.set_value(0.)

                # Get correct actions (supervision signal)
                acts_p =  get_acts(actions, prm.max_hops_train, k_beam=1)

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
                
                if uidx > 1 and prm.learning.lower() == 'q_learning':
                    # Randomly select experiences and convert them to numpy arrays.
                    idxs = np.random.choice(np.arange(len(experience)), size=len(queries))
                    rvs = []
                    for j in range(len(experience[idxs[0]])):
                        rv = []
                        for idx in idxs:
                            rv.append(experience[idx][j])

                        rvs.append(np.asarray(rv))
                else:
                    rvs = [np.zeros((len(queries),prm.max_words_query*prm.n_consec),dtype=np.float32), # rs_q
                           np.zeros((len(queries),prm.max_words_query*prm.n_consec),dtype=np.float32), # rs_q_m
                           np.zeros((len(queries),prm.max_hops_train+1),dtype=np.int32), # rl_idx
                           np.zeros((len(queries),prm.max_hops_train+1),dtype=np.float32), # rt
                           np.zeros((len(queries),prm.max_hops_train+1),dtype=np.float32) # rr
                          ]

                cost, R, l_idx, pages_idx, best_page_idx, best_answer, mask, dist \
                        = f_grad_shared(q_i, q_m, root_pages, acts_p, uidx, *rvs)
                f_update(prm.lrate)

                if prm.learning.lower() == 'q_learning': 
                    # update weights of the next_q_val network.
                    if (prm.update_freq > 0 and uidx % prm.update_freq == 0) or (uidx == prm.replay_start):
                        for tk, tv in tparams.items():
                            if tk in tparams_next:
                                tparams_next[tk].set_value(tv.get_value().copy())

                # Only update memory after freeze_mem or before replay_start.
                if (uidx < prm.replay_start or uidx > prm.freeze_mem) and prm.learning.lower() == 'q_learning':
                    # Update Replay Memory.
                    t = np.zeros((len(queries), prm.max_hops_train+1))
                    rR = np.zeros((len(queries), prm.max_hops_train+1))

                    for i in range(len(queries)):
                        j = np.minimum(mask[i].sum(), prm.max_hops_train)
                        # If the agent chooses to stop or the episode ends,
                        # the reward will be the reward obtained with the chosen document.
                        rR[i,j] = R[i]
                        t[i,j] = 1.
                        
                        add = True
                        if prm.selective_mem >= 0 and uidx > 1:
                            # Selective memory: keep the percentage of memories
                            # with reward=1 approximately equal to <selective_mem>.
                            pr = float(np.asarray(experience_r).sum()) / max(1., float(len(experience_r)))
                            if (pr < prm.selective_mem) ^ (rR[i,j] == 1.): # xor
                                add = False

                        if add:
                            experience.append([q_i[i], q_m[i], l_idx[i], t[i], rR[i]])
                            experience_r.append(rR[i])

                if np.isnan(cost) or np.isinf(cost):
                    print 'NaN detected'
                    return 1., 1., 1.
    
                #if uidx % 100 == 0:
                #    vis_att(pages_idx[:,-1], queries[-1], alpha[:,-1,:], uidx, options)

                if np.mod(uidx, prm.dispFreq) == 0:
                    with open(prm.outpath, "a") as fout:
                        fout.write("\n\nQuery: " + queries[-1].replace("\n"," "))
                        fout.write('\nBest Answer: ' + utils.idx2text(best_answer[-1], options['vocabinv']))
                        fout.write('\nBest page: ' + options['wiki'].get_article_title(best_page_idx[-1]))

                        for i, page_idx in enumerate(pages_idx[:,-1]):
                            fout.write('\niteration: ' +str(i) + " page idx " + str(page_idx) + ' title: ' + options['wiki'].get_article_title(page_idx))
                       
                        fout.write('\nEpoch '+ str(eidx) + ' Update '+ str(uidx) + ' Cost ' + str(cost) + \
                                   ' Reward Mean ' + str(R.mean()) + ' Reward Max ' + str(R.max()) + \
                                   ' Reward Min ' + str(R.min()) + \
                                   ' Q-Value Max (avg per sample) ' + str(dist.max(2).mean()) + \
                                   ' Q-Value Mean ' + str(dist.mean()))
                        #fout.write("\nCost Supervised: " + str(cost_sup))
                        #fout.write("\nCost RL: " + str(cost_RL))

                        fout.write("\nTime per Minibatch Update: " + str(time.time() - st))
                       

                if prm.saveto and np.mod(uidx, saveFreq) == 0:
                    print 'Saving...',

                    if best_p is not None:
                        params = best_p
                    else:
                        params = unzip(tparams)
                    np.savez(prm.saveto, history_errs=history_errs, **params)
                    pkl.dump(options, open('%s.pkl' % prm.saveto, 'wb'), -1)
                    print 'Done'

                if np.mod(uidx, validFreq) == 0 or uidx == 1:
                    if prm.visited_pages_path:
                        shuffle = False
                    else:
                        shuffle = True
                    kf_train = get_minibatches_idx(len(q_train), prm.batch_size_pred, shuffle=shuffle, max_samples=train_size)
                    kf_valid = get_minibatches_idx(len(q_valid), prm.batch_size_pred, shuffle=shuffle, max_samples=valid_size)
                    kf_test = get_minibatches_idx(len(q_test), prm.batch_size_pred, shuffle=shuffle, max_samples=test_size)

                    is_train.set_value(0.)
                    sup.set_value(0.) # supervised mode off
                    mixer.set_value(0) # no supervision
                    max_hops.set_value(prm.max_hops_pred)
                    k_beam.set_value(prm.k)

                    with open(prm.outpath, 'a') as fout:
                        fout.write('\n\nComputing Error Training Set')
                    train_err, train_R, train_accp, visited_pages_train = pred_error(f_pred, q_train, a_train, options, kf_train)

                    with open(prm.outpath, 'a') as fout:
                        fout.write('\n\nComputing Error Validation Set')
                    valid_err, valid_R, valid_accp, visited_pages_valid = pred_error(f_pred, q_valid, a_valid, options, kf_valid)

                    with open(prm.outpath, 'a') as fout:
                        fout.write('\n\nComputing Error Test Set')
                    test_err, test_R, test_accp, visited_pages_test = pred_error(f_pred, q_test, a_test, options, kf_test)

                    if prm.visited_pages_path:
                        pkl.dump([visited_pages_train, visited_pages_valid, visited_pages_test], open(prm.visited_pages_path, 'wb'))

                    history_errs.append([valid_err[-1], test_err[-1]])

                    if (uidx == 0 or
                        valid_err[-1] <= np.array(history_errs)[:,0].min()):

                        best_p = unzip(tparams)
                        bad_counter = 0

                    with open(prm.outpath, "a") as fout:
                        fout.write('\n[{per hop}, Avg] Train err ' + str(train_err) + '  Valid err ' + str(valid_err) + '  Test err ' + str(test_err))
                        fout.write('\n[{per hop}, Avg] Train R ' + str(train_R) + '  Valid R ' + str(valid_R) + '  Test R ' + str(test_R))
                        fout.write('\nAccuracy Page Actions   Train ' + str(train_accp) + '  Valid ' + str(valid_accp) + '  Test ' + str(test_accp))

                    if (len(history_errs) > prm.patience and
                        valid_err[-1] >= np.array(history_errs)[:-prm.patience,
                                                               0].min()):
                        bad_counter += 1
                        if bad_counter > prm.patience:
                            print 'Early Stop!'
                            estop = True
                            break

            with open(prm.outpath, "a") as fout:
                fout.write('\nSeen %d samples' % n_samples)

            if estop:
                break

    except KeyboardInterrupt:
        print "Training interupted"

    end_time = time.time()
    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)

    is_train.set_value(0.)
    sup.set_value(0.) # supervised mode off
    mixer.set_value(0) # no supervision
    max_hops.set_value(prm.max_hops_pred)
    k_beam.set_value(prm.k)

    kf_train_sorted = get_minibatches_idx(len(q_train), prm.batch_size_train)

    train_err, train_R, train_accp, visited_pages_train = pred_error(f_pred, q_train, a_train, options, kf_train_sorted)
    valid_err, valid_R, valid_accp, visited_pages_valid = pred_error(f_pred, q_valid, a_valid, options, kf_valid)
    test_err, test_R, test_accp, visited_pages_test = pred_error(f_pred, q_test, a_test, options, kf_test)

    with open(prm.outpath, "a") as fout:
        fout.write('\n[{per hop}, Avg] Train err ' + str(train_err) + '  Valid err ' + str(valid_err) + '  Test err ' + str(test_err))
        fout.write('\n[{per hop}, Avg] Train R ' + str(train_R) + '  Valid R ' + str(valid_R) + '  Test R ' + str(test_R))
        fout.write('\nAccuracy Page Actions   Train ' + str(train_accp) + '  Valid ' + str(valid_accp) + '  Test ' + str(test_accp))

    if prm.saveto:
        np.savez(prm.saveto, train_err=train_err,
                    valid_err=valid_err, test_err=test_err,
                    history_errs=history_errs, **best_p)
    with open(prm.outpath, "a") as fout:
        fout.write('\nThe code run for %d epochs, with %f sec/epochs' % ((eidx + 1), (end_time - start_time) / (1. * (eidx + 1))))
    with open(prm.outpath, "a") as fout:
        fout.write('\nTraining took %.1fs' % (end_time - start_time))
    return train_err, valid_err, test_err


if __name__ == '__main__':
    # See parameters.py for all possible parameter and their definitions.
    train_lstm()
