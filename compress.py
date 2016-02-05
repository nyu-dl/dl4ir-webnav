'''
Create a compressed hdf5 file from the orignal hdf5 file.
'''

import h5py
import os

in_path = '/scratch/rfn216/QA_wiki/data/wiki_171_24810.hdf5'
out_path = '/scratch/rfn216/QA_wiki/data/wiki_171_24810_compressed.hdf5'

f = h5py.File(in_path, 'r')

print 'Saving pages and links...'
os.remove(out_path) if os.path.exists(out_path) else None
fout = h5py.File(out_path,'w')

dt = h5py.special_dtype(vlen=bytes)
articles = fout.create_dataset("text", (len(f['text']),), dtype=dt, compression="gzip")
titles = fout.create_dataset("title", (len(f['title']),), dtype=dt, compression="gzip")
links = fout.create_dataset("links", (len(f['links']),), dtype=dt, compression="gzip")

for i in range(len(f['text'])):
    print 'processing article', i
    articles[i] = f['text'][i]
    titles[i] = f['title'][i]
    links[i] = f['links'][i]

f.close()
fout.close()
