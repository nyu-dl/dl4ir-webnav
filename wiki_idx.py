'''
Class to access the Wikipedia articles' word indexes stored in the hdf5 file.
'''
import h5py
import parameters as prm

class WikiIdx():

    def __init__(self, path):
        if prm.load_emb_mem:
            #self.f = h5py.File(path, 'r', driver='core')
            # don't use driver='core'. Reading from the numpy array
            # is faster for large number of indexes.
            ft = h5py.File(path, 'r')
            self.f = {}
            self.f['idx'] = ft['idx'].value
            if 'mask' in ft:
                self.f['mask'] = ft['mask'].value

        else:
            self.f = h5py.File(path, 'r')


    def get_article_idx(self, article_id):
        return self.f['idx'][article_id]


    def get_article_mask(self, article_id):
        return self.f['mask'][article_id]
