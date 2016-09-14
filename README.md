# WebNav

WebNav is a benchmark task for evaluating an agent with abilities to understand natural language and plan on partially observed environments. In this challenging task, an agent navigates through a web site consisting of web pages and hyperlinks to find a web page in which a query appears.

WebNav automatically transforms a website into this goal-driven web navigation task. As an example, we make WikiNav, a dataset constructed from the English Wikipedia containing approximately 5 million articles and more than 12 million queries for training. 

With this benchmark, we expect faster progress in developing artificial agents with natural language understanding and planning skills.

Paper: [End-to-End Goal-Driven Web Navigation](http://arxiv.org/pdf/1602.02261v1.pdf)

[Web Demo](http://webnav.cims.nyu.edu:10001/)


## WikiNav Dataset and Other Files

The WikiNav and WikiNav-Jeopardy datasets and auxiliary files can be [downloaded here](https://drive.google.com/folderview?id=0B5LbsF7OcHjqUFhWQ242bzdlTWc&usp=sharing):

* **wiki.hdf5**: English Wikipedia articles and hyperlinks (compiled from the [dump file of September/2015](https://dumps.wikimedia.org/enwiki/20150901/enwiki-20150901-pages-articles.xml.bz2)). In this dataset meta articles, whose titles start with "Wikipedia:", and articles with more than 300 hyperlinks are excluded. Any hyperlink that leads to a web page outside Wikipedia is removed in advance together with the following sections: "References", "External Links", "Bibliography" and "Partial Bibliography". Tables and figures are also removed.
* **wiki_emb.hdf5**: Articles' embeddings, computed as the average word vector representation of all the words in the article. This file is used to speed-up training by providing pre-computed embeddings.
* **queries_paths.zip**: queries (up to four sentences, randomly extracted from the articles) and paths (list of pages to be followed until the page that contains the query is reached).
* **queries_paths_jeopardy.hdf5**: Jeopardy! dataset: questions and answers pairs as well as the paths to the Wikipedia article whose title is the answer to the question.
* **model_d.npz**: model (d) trained on WikiNav-16-4: 8-Layer LSTM with 2048 units + Attention on queries.
* **D_cbow_pdw_8B.pkl**: a python dictionary containing 374,000 words where the values are pretrained embeddings from ["Word2Vec tool"](https://code.google.com/archive/p/word2vec/).
* **wiki_idf.pkl**: a python dictionary containing 374,000 words where the values are the Inverse Document Frequencies (IDF) computed from the English Wikipedia.
* **cat_pages.pkl**: a python dictionary where the keys are the Wikipedia's categories and the values are the lists of pages inside the categories.
* **page_pos.pkl**: a python dictionary where the keys are the articles' titles and the values are the page positions (offset in bytes) in the Wikipedia's dump file.
* **page_size.pkl**: a python dictionary where the keys are the articles' titles and the values are the page sizes (in bytes) in the Wikipedia's dump file.

## Accessing the Dataset

Due to their large sizes, the Wikipedia articles and queries files are stored in the HDF5 format,
which allows fast access without having to load them entirely into memory.

We provide wrapper classes (wiki.py and qp.py) to make your life easier when accessing these files.

For instance, the text and links of the "Machine Learning" article can be accessed using the Python code below (the h5py package is required):

```
import wiki
wk = wiki.Wiki('path/to/the/wiki.hdf5')

#get a dictionary of <article_title, article_id>
titles_ids = wk.get_titles_pos() 

# get the id of the 'Machine learning' article
article_id = titles_ids['Machine learning']

#get the article's text
text = wk.get_article_text(article_id)
pring 'text', text

#get links in this article as an list of articles' ids
links = wk.get_article_links(article_id)
print 'links:', links
```

You can also iterate over all pages:

```
for i, text in enumerate(wk.get_text_iter()):
    print 'content:', text
    print 'links:', wk.get_article_links(i)
    print 'title:', wk.get_article_title(i)
```


A sample code to access the queries and their paths starting from the page 'Category:Main_topic_classifications' is provided below:

```
import qp
qpp = qp.QP('path/to/the/queries_paths_2hops.hdf5')

# The input to get_queries() is a list containing the names
# of the sets you want to retrieve.
# Valid options are: 'train', 'valid', 'test'.

q_train = qpp.get_queries(['train'])  
p_train = qpp.get_paths(['train'])

print q_train[100] # get the query sample #100
print p_train[100] # get the path for query #100
```


##Creating Your Dataset

If you want to generate your own dataset from a website, run 'create_dataset.py'. Don't forget to change the paths inside the parameters.py file to point to where you want to save the dataset.

The file 'wiki_parser.py' contains the code to parse a Wikipedia article from the dump file, given its title. If you want to parse a particular website, you will have to create your own parser method. The only input to the parse() function is the URL of the web page, and it must return the content (as a string) and a list of hyperlinks inside the page.

A simple example of a parser class with no text cleaning is provided in simple_parser.py.



## Running the Model

After changing the properties in the parameters.py file to point to your local paths, the model can be trained using the following command:

```
THEANO_FLAGS='floatX=float32' python run.py
```

If you want to use a GPU:

```
THEANO_FLAGS='floatX=float32,device=gpu0,nvcc.fastmath=True' python run.py
```



##Dependencies

To run the code, you will need:
* Python 2.7
* [NumPy](http://www.numpy.org/)
* [scikit learn](http://scikit-learn.org/stable/index.html)
* [Theano](http://deeplearning.net/software/theano/)
* [NLTK](http://www.nltk.org/)
* [h5py](http://www.h5py.org/)

We recommend that you have at least 32GB of RAM. If you are going to use a GPU, the card must have at least 6GB.



## Reference

If you use this code as part of any published research, please acknowledge the
following paper:

**"End-to-End Goal-Driven Web Navigation"**  

Nogueira, Rodrigo and Cho, Kyunghyun

*To appear at NIPS (2016)*

    @inproceedings{nogueira2016end,
      title={End-to-End Goal-Driven Web Navigation},
      author={Nogueira, Rodrigo and Cho, Kyunghyun},
      booktitle={NIPS 2016},
      year={2016}
    }

## License

Copyright (c) 2016, Rodrigo Nogueira and Kyunghyun Cho

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of WebNav nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
