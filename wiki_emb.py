'''
Class to access the Wikipedia articles' embeddings stored in the hdf5 file.
'''
import h5py
import parameters as prm

class WikiEmb():

    def __init__(self, path):
        if prm.load_emb_mem:
            self.f = h5py.File(path, 'r', driver='core')
        else:
            self.f = h5py.File(path, 'r')


    def get_article_emb(self, article_id):
        return self.f['emb'][article_id]


    def get_article_mask(self, article_id):
        return self.f['mask'][article_id]
