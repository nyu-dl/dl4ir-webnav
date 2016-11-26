'''
For each <question, correct page> pair,
get paths from start page to the correct
page and append them to the dataset.
'''
import parameters as prm
import wiki
import csv
import h5py
import os
import re
import networkx as nx

def lreplace(pattern, sub, string):
    """
    Replaces 'pattern' in 'string' with 'sub' if 'pattern' starts 'string'.
    """
    return re.sub('^%s' % pattern, sub, string)

def rreplace(pattern, sub, string):
    """
    Replaces 'pattern' in 'string' with 'sub' if 'pattern' ends 'string'.
    """
    return re.sub('%s$' % pattern, sub, string)


def get_title(a, titles_pos):
    '''
    Find a Wikipedia title that match with Jeopardy Answer.
    '''

    r = a.lower()
    if r in titles_pos:
        return r
    else:
        r = lreplace('the\ ', '', r)
        if r in titles_pos:
            return r
        else:
            r = lreplace('a\ ', '', r)
            if r in titles_pos:
                return r
            else:
                r = r.replace('(', '').replace(')', '')
                if r in titles_pos:
                    return r
                else:
                    r = re.sub('\(.*\)', '', a.lower()).strip()
                    if r in titles_pos:
                        return r
    return ''


print 'Loading data...'

wk = wiki.Wiki(prm.pages_path)
titles_pos = wk.get_titles_pos()

print 'Creating child-parent dictionary...'
G = nx.DiGraph()

for i, (title, idd) in enumerate(titles_pos.items()):

    links = wk.get_article_links(idd) 
    for link in links:
        G.add_edge(idd, link, weight=1.)

    if i % prm.dispFreq == 0:
        print 'page', i

print 'Finding paths to answers...'

qat = [] # questions, answers, wikipedia page ids
qs = [] # questions only
with open(prm.jeopardy_path, 'rb') as csvfile:

    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    reader.next() # skip header
    i = 0
    for row in reader:
        q = row[3] + ' ' + row[5]
        a = row[6]
        t = get_title(a, titles_pos)
        if t != '':
            qat.append([q,a,t])
            qs.append(q)

        if i % prm.dispFreq == 0:
            print 'sample', i
        i += 1
 

print 'Finding candidate pages using the search engine...'

if str(prm.search_engine).lower() == 'google':
    import google_search
    candidates = google_search.get_candidates(qs)
if str(prm.search_engine).lower() == 'lucene':
    import lucene_search
    candidates = lucene_search.get_candidates(qs)
elif str(prm.search_engine).lower() == 'simple':
    import simple_search
    candidates = simple_search.get_candidates(qs)
elif not prm.search_engine:
    candidates = len(qs)*[[]]
else:
    raise ValueError('Not a valid value for the search engine.' + \
                     ' Valid options are: "google", "lucene", "simple", and None.')

print 'len(qat)', len(qat)
qatpc = []
for i,[q,a,t] in enumerate(qat):

    std_path = True
    print 'q:', q,
    print 't:', t
    if not prm.start_from_root:
        src = int(candidates[i][0])
        print 'src:', src,
        if len(wk.get_article_links(src)) > 0:
            path_idx = nx.shortest_path(G, source=src, target=titles_pos[t])
            if len(path_idx) <= prm.max_hops_pages + 1:
                std_path = False
    
    if std_path:
        path_idx = nx.shortest_path(G, source=titles_pos[prm.root_page], target=titles_pos[t])

    if len(path_idx) <= prm.max_hops_pages + 1:
        qatpc.append([q,a,t,path_idx,candidates[i]])

print 'len(qatpc)', len(qatpc)

print 'Dividing samples into training, validation and testing sets...'
qpc_all = []
n = 0
for n_max in prm.jeopardy_n_samples[::-1]:
    qpc = []
    nn=0    
    while (nn < n_max) and (n < len(qatpc)):
        q, a, t, p, c = qatpc[n]
        qpc.append([q, p, c])
        nn += 1
        n += 1
    qpc_all.append(qpc)

qpc_all = qpc_all[::-1]

print 'Saving queries and paths to HDF5...'
os.remove(prm.qp_path) if os.path.exists(prm.qp_path) else None
fout = h5py.File(prm.qp_path, 'w')
dt = h5py.special_dtype(vlen=bytes)

for i, set_name in enumerate(['train', 'valid', 'test']):
    qpc = qpc_all[i]
    print 'len(qpc)', len(qpc)

    dq = fout.create_dataset('queries_'+set_name, (len(qpc),), dtype=dt)
    dp = fout.create_dataset('paths_'+set_name, (len(qpc),), dtype=dt)
    dc = fout.create_dataset('candidates_'+set_name, (len(qpc),), dtype=dt)
    for j, (query, path, candidates) in enumerate(qpc):
        dq[j] = query
        dp[j] = ' '.join(str(x) for x in path)
        dc[j] = ' '.join(str(x) for x in candidates)

fout.close()
