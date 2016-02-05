'''
Simple parser that extracts a webpage's content and hyperlinks.
'''

import urllib2
import re

class Parser():
    
    def __init__(self):
        pass

    def parse(self, url):
        f = urllib2.urlopen(url)
        text = f.read()  # get page's contents.
 
        #use re.findall to get all the links
        links = re.findall('href=[\'"]?([^\'" >]+)', text)

        return text, links
