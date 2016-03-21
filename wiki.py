'''
Class to access wikipedia articles and links stored in the hdf5 file.
'''
import h5py

class Wiki():

    def __init__(self, path):
        self.f = h5py.File(path, 'r')


    def get_article_text(self, article_id):
        return self.f['text'][article_id]
 

    def get_article_title(self, article_id):
        return self.f['title'][article_id]


    def get_article_links(self, article_id):       
        links = self.f['links'][article_id].strip().split(' ')
        if links[0] != '':
            links = [int(i) for i in links] #convert to integer
        else:
            links = []
        return links


    def get_titles_pos(self):
        '''
        Return a dictionary where the keys are articles' titles and the values are their offset in the data array.
        '''
        return dict((el,i) for i,el in enumerate(self.f['title'].value))
       

    def get_text_iter(self):
        return self.f['text']


    def get_title_iter(self):
        return self.f['title']
