'''
Use Google Search API to retrieve candidate documents for given a query.
The API can be obtained at: https://pypi.python.org/pypi/google
'''

import google
import wiki
import parameters as prm

def get_candidates(qatp):

    wk = wiki.Wiki(prm.pages_path)
    titles_pos = wk.get_titles_pos()

    candidates = []
    n = 0
    for q,a,t,p in qatp:
        if n % 100 == 0:
            print 'finding candidates sample', n
        n+=1

        c = []

        for page in google.search(q.lower() + ' site:wikipedia.org', num=prm.max_candidates,stop=prm.max_candidates, pause=45):
            title = page.replace('https://en.wikipedia.org/wiki/','').replace('_',' ').lower()
            if title in titles_pos:
                c.append(titles_pos(title))

        candidates.append(c)
        
    return candidates
