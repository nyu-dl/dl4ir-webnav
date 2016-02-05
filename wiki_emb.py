'''
Class to access the Wikipedia articles' embeddings stored in the hdf5 file.
'''
import h5py

class WikiEmb():

    def __init__(self, path):
        #self.f = h5py.File(path, 'r', driver='core')
        self.f = h5py.File(path, 'r')


    def get_article_emb(self, article_id):
        return self.f['emb'][article_id]
