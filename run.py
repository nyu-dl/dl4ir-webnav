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
from collections import OrderedDict
import utils
from op_link import Link
from op_sentence import Sentence
from sklearn.decomposition import PCA
import wiki
import wiki_emb
import qp
import parameters as prm

# compute_test_value is 'off' by default, meaning this feature is inactive
#theano.config.compute_test_value = 'warn' # Use 'warn' to activate this feature

# Set the random number generators' seeds for consistency
SEED = 123
np.random.seed(SEED)


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


def normalize(x, stats_vars, mean_, std_, t_samples_, is_train):

    mean, std, t_samples = stats_vars['mean'], stats_vars['std'], stats_vars['t_samples']
    mean_batch = x.mean(0,keepdims=True)
    std_batch = x.std(0,keepdims=True)
    x_ = tensor.switch(tensor.eq(is_train, 1),
                       (x - mean_batch), #/ (std_batch + 0.1), # Small constant added to avoid large numbers)
                       (x - mean[None,:])) #/ (std[None,:] + 0.1))

    #update mean
    mean_ = (t_samples_ * mean_ + x.shape[0] * mean_batch[0,:]) / (t_samples_ + x.shape[0])
            
    #update std
    std_ = (t_samples_ * std_ + x.shape[0] * std_batch[0,:]) / (t_samples_ + x.shape[0])

    # update t_samples
    t_samples_ = t_samples_ + x.shape[0]

    return x_


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
    m = tensor.max(x, axis=1, keepdims=True)
    e_x = tensor.exp(x - m) * mask
    return e_x / (e_x.sum(axis=1, keepdims=True) + 1e-8) #this small constant avoids possible division by zero created by the mask


def init_params():
    params = OrderedDict()

    if prm.encoder.lower() == 'lstm':
        mul = 4
    else:
        mul = 1

    params['W'] = 0.01 * np.random.randn(prm.n_words, prm.dim_emb).astype(config.floatX) # vocab to word embeddings
    params['E_L'] = 0.01 * np.random.randn(prm.dim_emb, mul * prm.dim_proj).astype(config.floatX) # section to hidden state, link step
    params['E_Q'] = 0.01 * np.random.randn(prm.dim_emb, mul * prm.dim_proj).astype(config.floatX) # question to hidden state, link step
    params['U_I'] = 0.01 * np.random.randn(prm.dim_proj, mul * prm.dim_proj).astype(config.floatX) # transition link
    params['U_O'] = 0.01 * np.random.randn(prm.dim_proj, prm.dim_proj).astype(config.floatX) # hidden state link selection
    params['b'] = np.zeros((mul * prm.dim_proj,)).astype(config.floatX) # bias
    params['stop'] = 0.01 * np.random.randn(prm.dim_emb).astype(config.floatX) # stop action vector

    if prm.attention:
        params['E_A'] = 0.01 * np.random.randn(prm.dim_emb, 1).astype(config.floatX)
        params['U_A'] = 0.01 * np.random.randn(prm.dim_proj, 1).astype(config.floatX)
        params['L_A'] = 0.01 * np.random.randn(prm.dim_proj, 1).astype(config.floatX)
        params['b_A'] = np.zeros((1,)).astype(config.floatX) # bias

    if prm.idb:
        params['R_W'] = 0.01 * np.random.randn(prm.dim_proj, 1).astype(config.floatX) # question   
        params['R_b'] = np.zeros((1,)).astype(config.floatX) # bias
    
    exclude_params = {}
    if prm.fixed_wemb:
        exclude_params['W'] = True
    
    return params, exclude_params


def adam(lr0, tparams, grads, tq, tq_m, troot_pages, acts_p, cost,
         scan_updates, baseline_updates, stats_updates, opt_out=[]):
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k) for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    
    
    f_grad_shared = theano.function([tq, tq_m, troot_pages, acts_p], [cost]+opt_out, 
                                    updates=gsup+scan_updates+baseline_updates+stats_updates)

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


def build_model(tparams, baseline_vars, stats_vars, options):
    trng = RandomStreams(SEED)
    off = 1e-8  # small constant to avoid log 0 = -inf
    consider_constant = []

    is_train = theano.shared(np_floatX(0.)) # Used for dropout.
    sup = theano.shared(np_floatX(0.)) # Supervised or not
    max_hops = theano.shared(np.asarray(0, dtype=np.int32)) # Max number of iterations
    k_beam = theano.shared(np.asarray(0, dtype=np.int32)) # top-k items in the beam search.

    mean_ = tensor.alloc(np.array(0., dtype=np.float32), prm.dim_proj)
    std_ = tensor.alloc(np.array(0., dtype=np.float32), prm.dim_proj)
    t_samples_ = tensor.alloc(np.array(0., dtype=np.float32), 1)

    q = tensor.imatrix('q')
    q_m = tensor.fmatrix('q_m')
    root_pages = tensor.fvector('root_pages')
    acts_p = tensor.fmatrix('acts_p')

    #q.tag.test_value = np.zeros((prm.batch_size_train,prm.n_consec*prm.max_words_query), dtype='int32')
    #q_m.tag.test_value = np.ones((prm.batch_size_train,prm.n_consec*prm.max_words_query), dtype=theano.config.floatX)
    #root_pages.tag.test_value = np.zeros((prm.batch_size_train,), dtype=theano.config.floatX)
    #acts_p.tag.test_value = np.zeros((prm.max_hops_train+1,prm.batch_size_train), dtype=theano.config.floatX)

    n_samples = q.shape[0]

    get_links = Link(options['wiki'], options['wikiemb'], options['vocab'], prm.dim_emb) # instantiate custom function to get wiki links

    def _step(it, act_p, p_, m_, l_a_, h_, c_, q_a):

        if prm.attention:
            e = tensor.dot(q_a, tparams['E_A'])[:,:,0]
            e += tensor.dot(h_, tparams['U_A'])[:,0][:,None]
            e += tensor.dot(l_a_, tparams['L_A'])[:,0][:,None]
            e += tparams['b_A'][None,0][:,None]
            # repeat for beam search
            q_m_ = tensor.extra_ops.repeat(q_m, k_beam, axis=0)
            alpha = softmax_mask(e, q_m_)
            q_at = (alpha[:,:,None] * q_a).sum(1)
        else:
            q_at = q_a

        L_a, L_m, l_page, l_truth = get_links(p_, act_p, k_beam, it)

        # Select hyperlink from section
        preact = tensor.dot(h_, tparams['U_I'])
        preact += tensor.dot(q_at, tparams['E_Q'])
        preact += tensor.dot(l_a_, tparams['E_L'])
        preact += tparams['b']

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


        if prm.normalize:
            L_a = normalize(L_a, stats_vars,  mean_, std_, t_samples_, is_train)

        n_links = L_a.shape[1]
        
        # append stop action
        stop = tparams['stop'][None, None, :] # stop.shape = (1, 1, dim_emb)
        stop = tensor.extra_ops.repeat(x=stop, repeats=n_samples * k_beam, axis=0)  # stop.shape = (n_samples, 1, dim_emb)
        L_a = tensor.concatenate([L_a, stop], axis=1)  #L_a.shape = (n_samples, n_links+1, dim_emb)
        stop_m = tensor.alloc(np_floatX(1.), n_samples*k_beam, 1)
        L_m = tensor.concatenate([L_m, stop_m], axis=1)  #L_m.shape = (n_samples, n_sections+1)
        n_links_ = n_links + 1


        #get the distribution for link action
        z = tensor.tanh(tensor.dot(h, tparams['U_O']))   # h=(n_samples, dim_proj)  U_O=(dim_proj, dim_emb)    z=(n_samples,dim_emb)
        z = tensor.extra_ops.repeat(z.reshape((z.shape[0], 1, z.shape[1])), repeats=n_links_, axis=1)  # z=(n_samples,n_links_,dim_emb)
        res = (L_a * z).sum(axis=2)   # res=(n_samples * k_beam, n_links_)
        res = res.reshape((n_samples, k_beam * (n_links_))) # Reshape for beam search
        L_m = L_m.reshape((n_samples, k_beam * (n_links_)))
        dist = softmax_mask(res, L_m)

        # Select hyperlink by sampling from the distribution.
        lp_ = tensor.eq(is_train,1.).astype('float32') * trng.multinomial(n=1, pvals=dist, dtype=dist.dtype) + (1. - tensor.eq(is_train,1.).astype('float32')) * dist

        res1 = tensor.eq(it, 0.).astype('int32') * lp_[:,:n_links_].argsort(axis=1)[:, -k_beam:].flatten().astype("int32") \
               + (1 - tensor.eq(it, 0.).astype('int32')) * lp_.argsort(axis=1)[:, -k_beam:].reshape((n_samples * k_beam,)).astype("int32")

        # Select action: supervised mode or not.
        l_idx = (sup * l_truth + (1 - sup) * res1).astype("int32")

        l_idx0 = (k_beam * tensor.floor(tensor.arange(l_idx.shape[0]) / k_beam)  + tensor.floor(l_idx / (n_links_)) ).astype('int32')
        l_idx1 = tensor.mod(l_idx, n_links_)
                
        dist = dist.reshape((n_samples*k_beam, n_links_))
        l_prob = dist[l_idx0, l_idx1] # get the probability of the chosen action.
        l_ent = -(dist * tensor.log(dist + off)).sum(1) # get the entropy.

        #supervised approach: compute the cost for page selection
        cost_p = tensor.eq(sup,1.).astype('float32') * -tensor.log(dist[tensor.arange(dist.shape[0]), l_truth] + off)

        m = 1.0 - tensor.eq(l_idx1, n_links).astype("float32")

        # update the mask vector for the coming iterations.
        m = m * m_ #consider previous mask
            
        # Apply mask to not consider the stop action by setting l_idx0=0 and l_idx1=0
        l_idx0_ = l_idx0 * m.astype("int32")
        l_idx1_ = l_idx1 * m.astype("int32")

        # Get indices of the next articles.
        p = l_page[l_idx0_, l_idx1_]
        p = p * m  - (1. - m)  # Set next page idx to -1 in case stop action was chosen. 
        
        l_a = L_a[l_idx0_, l_idx1_, :]

        return p, m, l_a, h, c, l_prob, l_ent, cost_p

    #get embeddings for the queries
    q_a = (tparams['W'][q.flatten()].reshape((q.shape[0], q.shape[1], prm.dim_emb)) * q_m.reshape((q_m.shape[0], q_m.shape[1], 1)))

    if not prm.attention:
        q_a = q_a.sum(1)

    q_a = q_a.astype('float32')

    if prm.normalize:
        q_a = normalize(q_a, stats_vars,  mean_, std_, t_samples_, is_train)

    #repeat question for beam search
    q_a = tensor.extra_ops.repeat(q_a, k_beam, axis=0)
    root_pages_ = tensor.extra_ops.repeat(root_pages, k_beam)

    (page_idx, mask, _, h, _, l_prob, l_ent, cost_p), scan_updates = theano.scan(_step,
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
                                              ],
                                non_sequences=[q_a],
                                name='lstm_layers',
                                n_steps=max_hops+1)


    # Get only the used probabilities.
    mask_ = tensor.concatenate([tensor.alloc(np.array(1.), 1, mask.shape[1]), mask], axis=0)[:-1,:]
    l_prob *= mask_   # l_prob.shape = (n_iterations, n_samples)
    l_ent *= mask_   # l_ent.shape = (n_iterations, n_samples)

    # get last valid action before the stop action. In case the mask is all marked, get the last action.
    j = (tensor.minimum(tensor.alloc(mask.shape[0]- 1, mask.shape[1]), mask.sum(axis=0))).astype("int32") 

    get_sent = Sentence(options['wiki'], options['vocab'], prm.n_consec) # instantiate custom function to get sentences 
    page_idx_ = tensor.concatenate([root_pages_[None,:], page_idx],axis=0)

    pps = page_idx_.T.flatten()

    R, best_page_idx, best_answer = get_sent(q, q_m, pps, k_beam*(max_hops+2))
    f_pred = theano.function([q, q_m, root_pages, acts_p], [best_answer, best_page_idx, R, page_idx, ], updates=scan_updates, name='f_pred', on_unused_input='ignore')

    if prm.normalize:
        stats_updates = [(stats_vars['mean'], (stats_vars['mean'] * stats_vars['t_samples'][0] + mean_ * t_samples_[0]) / (stats_vars['t_samples'][0] + t_samples_[0])),
                         (stats_vars['std'], (stats_vars['std'] * stats_vars['t_samples'][0] + std_ * t_samples_[0]) / (stats_vars['t_samples'][0] + t_samples_[0])),
                         (stats_vars['t_samples'], stats_vars['t_samples'] + t_samples_)]
    else:
        stats_updates = []

    # entropy regularization
    cost_ent = -prm.erate * l_ent
    
    if prm.supervised:
        cost = ((cost_p + cost_ent) * mask_).sum(0).mean()
        baseline_updates = []
    else:
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

        cost = ( ( R_ * -tensor.log(tensor.maximum(l_prob, 0.1)) + cost_ent) * mask_).sum(0).mean() #Clipping l_prob so -log do not become too large.

        if prm.idb:
            R0 = R[:,None] - R_mean_
            consider_constant += [R0]
            cost += 0.01 * ((R_idb - R0) ** 2).mean()

    opt_out = {'R':R, 'page_idx':page_idx, 'best_answer':best_answer, 'best_page_idx':best_page_idx}

    return is_train, sup, max_hops, k_beam, q, q_m, root_pages, acts_p, f_pred, cost, scan_updates, baseline_updates, stats_updates, consider_constant, opt_out


def get_root_pages(actions):
    root_pages = np.zeros((len(actions)), dtype=np.float32)
    for t, action in enumerate(actions):
        root_pages[t] = action[0]
    return root_pages


def get_acts(actions, max_hops, k_beam):
    # Get correct actions (supervision signal)
    acts_p = -np.ones((max_hops+1, len(actions)), dtype=np.float32)
    for t, action in enumerate(actions):
        for kj, titlU_Id in enumerate(action[1:]):
            acts_p[kj, t] = titlU_Id

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

    for _, valid_index in iterator:
        q_bow, q_m = utils.BOW2([queries[t].lower() for t in valid_index], options['vocab'], prm.max_words_query*prm.n_consec)
        acts = [actions[t] for t in valid_index]

        #fake acts that won't be used in the prediction
        acts_p = -np.ones((prm.max_hops_pred+1, len(q_bow) * prm.k), dtype=np.float32)
        
        root_pages = get_root_pages(acts)

        best_answer, best_page_idx, R, pages_idx = f_pred(q_bow, q_m, root_pages, acts_p)
        R_binary = np.ones_like(R)
        R_binary[R<1.0] = 0.0
        n[-1] += len(valid_index)
        valid_R[-1] += R.sum()
        valid_acc[-1] += R_binary.sum()
        
        # get correct page-actions.
        acts_p = get_acts(acts, prm.max_hops_pred, prm.k)

        # Check how many page and section actions the model got right.
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

    valid_R = valid_R / n
    valid_err = 1 - valid_acc / n
    acts_pc = acts_pc / acts_pt

    return valid_err, valid_R, acts_pc


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
    qpp = qp.QP(prm.qp_path)
    q_train, q_valid, q_test = qpp.get_queries()
    a_train, a_valid, a_test = qpp.get_paths()

    print 'Building model'
    # This create the initial parameters as np ndarrays.
    # Dict name (string) -> np ndarray
    params, exclude_params = init_params()

    if prm.reload_model:
        load_params(prm.reload_model, params)

    if prm.wordemb_path:
        print 'loading pre-trained weights for word embeddings'
        params = load_wemb(params, vocab)
        options['W'] = params['W']

    # This create Theano Shared Variable from the parameters.
    # Dict name (string) -> Theano Tensor Shared Variable
    # params and tparams have different copy of the weights.
    tparams = init_tparams(params)

    mean = theano.shared(np.zeros((prm.dim_proj,)).astype(config.floatX)) # avg of the training set
    std = theano.shared(np.zeros((prm.dim_proj,)).astype(config.floatX)) # std of the training set
    t_samples = theano.shared(np.zeros((1,)).astype(config.floatX)) # total number of samples so far
    stats_vars = {'mean': mean, 'std': std, 't_samples': t_samples}
    
    if prm.supervised:
        baseline_vars = {}
    else:
        R_mean = theano.shared(0.71*np.ones((1,)), name='R_mean')
        R_std = theano.shared(np.ones((1,)), name='R_std')
        baseline_vars = {'R_mean': R_mean, 'R_std': R_std}


    is_train, sup, max_hops, k_beam, tq, tq_m, troot_pages, tacts_p, f_pred, cost, \
            scan_updates, baseline_updates, stats_updates, consider_constant, \
            opt_out = \
            build_model(tparams, baseline_vars, stats_vars, options)
            
            
    if prm.decay_c > 0.:
        decay_c = theano.shared(np_floatX(prm.decay_c), name='decay_c')
        weight_decay = 0.
        weight_decay += (tparams['U'] ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay
    
    #get only parameters that are not in the exclude_params list
    tparams_ = OrderedDict([(kk, vv) for kk, vv in tparams.iteritems() if kk not in exclude_params])

    grads = tensor.grad(cost, wrt=itemlist(tparams_), consider_constant=consider_constant)

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = optimizer(lr, tparams_, grads, tq, tq_m, troot_pages, tacts_p, cost, scan_updates, baseline_updates, \
                                       stats_updates, opt_out=[opt_out['R'], opt_out['page_idx'], opt_out['best_answer'], opt_out['best_page_idx']])

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
                
                if prm.supervised == 1:
                    sup_ = True
                elif prm.supervised > 1:
                    if uidx % (int(uidx / prm.supervised) + 1) == 0:
                        sup_ = True
                    else: 
                        sup_ = False
                else:
                    sup_ = False
                    
                if sup_:
                    sup.set_value(1.) # select supervised mode
                    # Get correct actions (supervision signal)
                    acts_p =  get_acts(actions, prm.max_hops_train, k_beam=1)
                else:
                    sup.set_value(0.) # select non-supervised mode
                    acts_p = -np.ones((prm.max_hops_train+1, len(queries)), dtype=np.float32)

                root_pages = get_root_pages(actions)
                
                # Get the BoW for the queries
                q_bow, q_m = utils.BOW2(queries, vocab, prm.max_words_query*prm.n_consec)
                n_samples += len(queries)
                cost, R, pagesidx, best_answer, best_page_idx = f_grad_shared(q_bow, q_m, root_pages, acts_p)
                f_update(prm.lrate) 
                if np.isnan(cost) or np.isinf(cost):
                    print 'NaN detected'
                    return 1., 1., 1.

                if np.mod(uidx, prm.dispFreq) == 0:
                    with open(prm.outpath, "a") as fout:
                        fout.write("\n\nQuery: " + queries[-1].replace("\n"," "))
                        fout.write('\nBest Answer: ' + utils.idx2text(best_answer[-1], options['vocabinv']))
                        fout.write('\nBest page: ' + options['wiki'].get_article_title(best_page_idx[-1]))

                        for i, pageidx in enumerate(pagesidx[:,-1]):
                            fout.write('\niteration: ' +str(i) + " page idx " + str(pageidx) + ' title: ' + options['wiki'].get_article_title(pageidx))
                       
                        fout.write('\nEpoch '+ str(eidx) + ' Update '+ str(uidx) + ' Cost ' + str(cost) + \
                                   ' Reward Mean ' + str(R.mean()) + ' Reward Max ' + str(R.max()) +  \
                                   ' Reward Min ' + str(R.min()))

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

                if np.mod(uidx, validFreq) == 0:

                    kf_train = get_minibatches_idx(len(q_train), prm.batch_size_pred, shuffle=True, max_samples=train_size)
                    kf_valid = get_minibatches_idx(len(q_valid), prm.batch_size_pred, shuffle=True, max_samples=valid_size)
                    kf_test = get_minibatches_idx(len(q_test), prm.batch_size_pred, shuffle=True, max_samples=test_size)

                    is_train.set_value(0.)
                    sup.set_value(0.) # supervised mode off
                    max_hops.set_value(prm.max_hops_pred)
                    k_beam.set_value(prm.k)

                    with open(prm.outpath, 'a') as fout:
                        fout.write('\n\nComputing Error Training Set')
                    train_err, train_R, train_accp = pred_error(f_pred, q_train, a_train, options, kf_train)

                    with open(prm.outpath, 'a') as fout:
                        fout.write('\n\nComputing Error Validation Set')
                    valid_err, valid_R, valid_accp = pred_error(f_pred, q_valid, a_valid, options, kf_valid)

                    with open(prm.outpath, 'a') as fout:
                        fout.write('\n\nComputing Error Test Set')
                    test_err, test_R, test_accp = pred_error(f_pred, q_test, a_test, options, kf_test)

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
    max_hops.set_value(prm.max_hops_pred)
    k_beam.set_value(prm.k)

    kf_train_sorted = get_minibatches_idx(len(q_train), prm.batch_size_train)

    train_err, train_R, train_accp = pred_error(f_pred, q_train, a_train, options, kf_train_sorted)
    valid_err, valid_R, valid_accp = pred_error(f_pred, q_valid, a_valid, options, kf_valid)
    test_err, test_R, test_accp = pred_error(f_pred, q_test, a_test, options, kf_test)

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
