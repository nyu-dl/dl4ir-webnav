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
        'dset': list of datasets to be returned ('train', 'valid' and/or 'test').
        '''
        outs = []
        for name in dset:
            outs.append(self.f['queries_'+name][:])

        return outs  


    def get_paths(self, dset=['train', 'valid', 'test']):
        '''
        Return the paths (as a list of articles' ids) to reach the query,
        starting from the root page.
        'dset': list of datasets to be returned ('train', 'valid' and/or 'test').
        '''
        outss = []
        for name in dset:
            outs = []
            for paths in self.f['paths_'+name]:
                out = []
                for path in paths.split('|'):
                    out.append(self.tolist(path))
                outs.append(out)
            outss.append(outs)

        return outss


    def get_candidates(self, dset=['train', 'valid', 'test']):
        '''
        Return the candidates (as a list of articles' ids) pages,
        selected by the reverse index search engine.
        'dset': list of datasets to be returned ('train', 'valid' and/or 'test').
        '''
        outs = []
        for name in dset:
            if 'candidates_'+name in self.f:
                out = []
                for item in self.f['candidates_'+name]:
                    out.append(self.tolist(item))
            else:
                out = len(self.f['queries_'+name])*[[]]
            outs.append(out)

        return outs


    def tolist(self, text):
        '''
        Convert a string whose elements are separated by a space to a list of integers.
        '''
        if len(text) > 0:
            return [int(a) for a in text.strip().split(' ')]
        else:
            return []
