'''
Compute the Inverse Document Frequency (IDF) of Wikipedia
articles using the vocabulary defined in <vocab_path>.
'''

import cPickle as pkl
import numpy as np
import random
import utils
from collections import OrderedDict
from nltk.tokenize import wordpunct_tokenize
import re
import parameters as prm

print 'loading vocabulary'
vocab = utils.load_vocab(prm.vocab_path, prm.n_words)

textbegin = False
title = ''
text = ''
n = 0
f = open(prm.dump_path, "rb")

print 'creating IDF'
m = 0 # number of documents
df = {}  # word-document frenquency

while True:
    line = f.readline()

    if (line == ''):
        break
    line = line.lower()

    if ("<page>" in line):
        pagebegin = True
        title = ""
        sections = []

    if ("</page>" in line):
        pagebegin = False
        title = ""

    if ("<title>" in line) and ("</title>" in line) and pagebegin:
        title = line.replace("    <title>","").replace("</title>\n","")

        if n % 100000 == 0:
            print n
        n += 1

        #if n > 1000000:
        #    break
    

    if line.strip()[:2] == "==" and line.strip()[-2:] == "==" and line.strip()[:3] != "===" and line.strip()[-3:] != "===": # another section begins...
        sections.append({"text": ""})
        
    if textbegin:
        if "[[category:" != line[:11]: # skip the categories
            line_clean = line.replace("</text>", "")
            text += line_clean
            sections[-1]["text"] += line_clean

    if ("<text xml:space=\"preserve\">" in line) and pagebegin:
        textbegin = True
        line_clean = line.replace("      <text xml:space=\"preserve\">","")
        sections.append({"text": line_clean}) #add a section, it will be the abstract
        text = line_clean

    if ("</text>" in line) and pagebegin:

            textbegin = False
            if "[[category:" != line[:11]: # skip the categories
                text += line.replace("</text>","")
            m += 1

            words_sections = []
            # Clean text and get hyperlinks
            for j, section in enumerate(sections):
                text = section["text"]
                text = text.replace('\n', ' ')
                text = re.sub(r'\&lt\;ref.*?\&lt\;\/ref\&g', '', text)
                text = re.sub(r'\&lt\;ref.*?\/\&gt\;', '', text)
                text = re.sub(r'\[\[file\:.*?\]\]', '', text)
                text = re.sub(r'\[\[image\:.*?\]\]', '', text)
                words = wordpunct_tokenize(text)
                words_sections += words


            # compute document frequency per word in the training set
            wb = dict.fromkeys(words_sections, 0) #remove duplicated words
            for w in wb.keys():
                if w in vocab:
                    if w not in df:
                        df[w] = 0
                    df[w] += 1

#compute inverse document frequency:
idf = dict.fromkeys(range(len(vocab)), 0) #initialize dic with length of vocabulary and values equal to 1.
for wi, fr in df.items():
    idf[wi] = np.log(float(m) / (1. + float(fr))) # total number of documents divided by the number of documents word w appears. Sum 1 to avoid division by zero

#normalize
maxidf = np.asarray(idf.values()).max()
for wi, fr in idf.items():
    idf[wi] = fr / maxidf

with open(prm.idf_path, "wb") as f:
    pkl.dump(idf, f)

f.close()

print 'done'
