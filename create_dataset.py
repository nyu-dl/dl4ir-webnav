'''
Create dataset from a website.
The default parser is wiki_parser, but you can create your own parser.

'''
import cPickle as pkl
import numpy as np
import utils
from nltk.tokenize import wordpunct_tokenize
import nltk.data
import h5py
import os
import convert2emb
import time
import wiki_parser as parser
import parameters as prm

redirects = {} # Dic of pages and their redirections.

print "loading class..."
ps = parser.Parser(redirects)

print 'loading vocabulary...'
vocab = utils.load_vocab(prm.vocab_path, prm.n_words)

print 'loading IDF dictionary...'
with open(prm.idf_path, "rb") as f:
    idf = pkl.load(f)

print "creating datasets..."
qp_all = {}  #stores queries and paths from all pages
pages = {}
queries_all = []
paths_all = []
nltk.download('punkt')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

n = 0

hops = 0
next_pages = [prm.root_page]
par_pages = {prm.root_page: []}
par_secs = {prm.root_page: []}
st = time.time()
while hops <= prm.max_hops_pages:
    curr_pages = list(next_pages)
    next_pages = []
    
    for title in curr_pages:
        if title not in pages:
            st = time.time()
            article = ps.parse(title)
            # if parse was successfull:
            if article:
                text, links = article

                if prm.max_links:
                    if len(links) > prm.max_links:
                        continue

          pages[title] = {}
          pages[title]['text'] = text
          pages[title]['links'] = links
          #print 'time total', (time.time() - st), ' time inside', (time.time() - st1), ' time recursive', (st3 - st2)

          log_txt = 'hops: ' + str(hops) + ' page num: ' + str(len(pages)) 
          with open('out2.log', 'a') as ftemp:
              ftemp.write(log_txt+'\n')
          print log_txt

        if title in pages: #if the page was added to the pages, then get some sample queries and next page

            for link in pages[title]['links']:
                if link not in par_pages: # Check if the page was already visited in this pages.
                    next_pages.append(link)
                    # Get the parent page for the supervision signal.
                    par_pages[link] = par_pages[title] + [title]
                       
            # only get queries up to max_hops. Higher hops will be used to get other pages only.
            if hops <= np.asarray(prm.max_hops).max():
                if ("category:" not in title) and (hops >= 1): # do not chose queries from categories or if it less than one hops
                    # compute TF
                    tf = utils.compute_tf(wordpunct_tokenize(pages[title]['text'].decode('ascii', 'ignore')), vocab)
                    # Get sentences
                    sents_pre = tokenizer.tokenize(pages[title]['text'].decode('ascii', 'ignore'))
                    sents = []
                    n_consec = min(len(sents_pre), prm.n_consec)
                    for sk in range(0,len(sents_pre)-n_consec+1):
                        sent = ''
                        for sj in range(n_consec):
                            sent += ' ' + sents_pre[sk+sj]
                        sents.append(sent.strip())
                    sents_filtered = []
                    for sent in sents:
                        n_words_sent = utils.n_words(wordpunct_tokenize(sent.lower()), vocab)
                        if n_words_sent >= prm.min_words_query and n_words_sent <= prm.n_consec*prm.max_words_query: #only add if the sentence has between 10 known and 30 known words
                            sents_filtered.append(sent)

                    # randomly select up to MAX_SENT sentences in this page.
                    if len(sents_filtered) > 0:
                        # Compute TFIDF score for each sentence
                        scores = np.zeros((len(sents_filtered)), dtype=np.float32)
                        for kk, sent in enumerate(sents_filtered):
                            tt = 0
                            for word in wordpunct_tokenize(sent.lower()):
                                if (word in idf) and (word in tf):
                                    scores[kk] += tf[word]*idf[word]
                                    tt += 1
                            scores[kk] = scores[kk] / tt
                        sents_idx = scores.argsort()[-prm.max_sents:]
                        qp_all[title] = []
                        for sent_idx in sents_idx:
                            sent = sents_filtered[sent_idx]
                            qp_all[title].append([sent, par_pages[title] + [title]])
                            n += 1
                            log_txt = 'hops: ' + str(hops) + ' sample num: ' + str(n)
                            with open('out2.log', 'a') as ftemp:
                                ftemp.write(log_txt +'\n')
                            print log_txt

    hops += 1

f.close()

# Apply link redirections.
for title, article in pages.items():
    for i,link in enumerate(article['links']):
        if link in redirects:
            article['links'].pop(i)
            article['links'].insert(i,redirects[link])

for page_qp in qp_all.values():
    for _, path in page_qp:
        for i, item in enumerate(path):
            if title in redirects:
                path.pop(i)
                path.insert(i,redirects[link])

for paths in paths_all:
    for path in paths:
        for i, item in enumerate(path):
            if title in redirects:
                path.pop(i)
                path.insert(i,redirects[link])


# Save pages to HDF5
print 'Saving text and links...'
os.remove(prm.pages_path) if os.path.exists(prm.pages_path) else None
fout = h5py.File(prm.pages_path,'w')

dt = h5py.special_dtype(vlen=bytes)
articles = fout.create_dataset("text", (len(pages),), dtype=dt)
titles = fout.create_dataset("title", (len(pages),), dtype=dt)
i=0
n_links = 0
title_idx = {}
for title, article in pages.items():
    articles[i] =  article['text']
    titles[i] =  title
    title_idx[title] = i
    i += 1
    n_links += len(article['text'])

links = fout.create_dataset("links", (n_links,), dtype=dt)
for i, article in enumerate(pages.values()):
    links_txt = ''
    for link in article['links']:
        if link in title_idx:
            links_txt += str(title_idx[link]) + ' '
    links[i] = links_txt.strip()

fout.close()


print 'Saving query sentences...'
lst_qp = qp_all.values()
for mm, max_hop in enumerate(prm.max_hops):

    queries_all = []
    paths_all = []
    n = 0
    for n_max in prm.n_samples[mm][::-1]:
        queries = []
        paths = []
        nn=0
        while (nn < n_max) and (n < len(lst_qp)):
            qqpp = lst_qp[n]
            for query, path in qqpp:
                if len(path)-1 <= max_hop:
                    queries.append(query)
                    paths.append(path)
                    nn += 1
            n += 1
            
        queries_all.append(queries)
        paths_all.append(paths)
    queries_all = queries_all[::-1]
    paths_all = paths_all[::-1]


    # Save the queries and paths to HDF5
    qp_path_mod = prm.qp_path_pre.replace('.hdf5','') + '_' +str(max_hop) + 'hops.hdf5'
    os.remove(qp_path_mod) if os.path.exists(qp_path_mod) else None
    fout = h5py.File(qp_path_mod,'w')

    # Write queries to file
    for i, set_name in enumerate(['train', 'valid', 'test']):    
        queries = queries_all[i]
        ds = fout.create_dataset('queries_'+set_name, (len(queries),), dtype=dt)
        for j, query in enumerate(queries):
            ds[j] = query

    # Convert title string to indexes in the path and write them to hdf5 file.
    for i, set_name in enumerate(['train', 'valid', 'test']):    
        paths_set = paths_all[i]
        ds = fout.create_dataset('paths_'+set_name, (len(paths_set),), dtype=dt)
        for j, paths in enumerate(paths_set):
            paths_ = ''
            for title in paths:
                paths_ += str(title_idx[title]) + ' '
            ds[j] = paths_.strip()

    fout.close()


if prm.pages_emb_path:
    print 'Computing embeddings...'
    convert2emb.compute_emb(prm.pages_path, prm.pages_emb_path, vocab)

print 'done'
