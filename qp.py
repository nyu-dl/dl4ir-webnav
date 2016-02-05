'''
Class to access queries and paths stored in the hdf5 file.
'''
import h5py

class QP():

    def __init__(self, path):
        self.f = h5py.File(path, 'r')


    def get_queries(self, dset=['train', 'valid', 'test']):
        '''
        Return the queries.
        'dset' is the list of datasets to be returned ('train', 'valid' and/or 'test').
        '''
        outs = []
        for name in dset:
            outs.append(self.f['queries_'+name][:])

        return outs  


    def get_paths(self, dset=['train', 'valid', 'test']):
        '''
        Return the paths (as a list of articles' ids) to reach the query,
        starting from the root page.
        'dset' is the list of datasets to be returned ('train', 'valid' and/or 'test').
        '''
        outs = []
        for name in dset:
            out = []
            for item in self.f['paths_'+name]:
                out.append(self.tolist(item))
            outs.append(out)

        return outs


    def tolist(self, text):
        '''
        Convert a string whose elements are separated by a space to a list of integers.
        '''
        return [int(a) for a in text.strip().split(' ')]
