# WebNav

WebNav is a benchmark task for evaluating an agent with abilities to understand natural language and plan on partially observed environments. In this challenging task, an agent navigates through a web site consisting of web pages and hyperlinks to find a web page in which a query appears.

WebNav automatically transforms a website into this goal-driven web navigation task. As an example, we make WikiNav, a dataset constructed from the English Wikipedia containing approximately 5 million articles and more than 12 million queries for training. 

With this benchmark, we expect faster progress in developing artificial agents with natural language understanding and planning skills.

Link to the paper: [WebNav: A New Large-Scale Task for Natural Language based Sequential Decision Making](http://arxiv.org/pdf/1602.02261v1.pdf)


## WikiNav Dataset and Other Files

The WikiNav dataset and auxiliary files can be [downloaded here](https://drive.google.com/folderview?id=0B5LbsF7OcHjqUFhWQ242bzdlTWc&usp=sharing):

* **wiki.hdf5**: English Wikipedia articles and hyperlinks (compiled from the [dump file of September/2015](https://dumps.wikimedia.org/enwiki/20150901/enwiki-20150901-pages-articles.xml.bz2)). In this dataset meta articles, whose titles start with "Wikipedia:", and articles with more than 300 hyperlinks were excluded. Any hyperlink that leads to a web page outside Wikipedia is removed in advance together with the following sections: "References", "External Links", "Bibliography" and "Partial Bibliography". Tables and figures were also removed.
* **wiki_emb.hdf5**: Articles' embeddings, computed as the average word vector representation of all the words in the article. This file is used to speed-up training by providing pre-computed embeddings.
* **queries_paths.zip**: queries (up to four sentences, randomly extracted from the articles) and paths (list of pages to be followed until the page that contains the query is reached).
* **D_cbow_pdw_8B.pkl**: a python dictionary containing 374,000 words where the values are pretrained embeddings as in ["Learning to Understand Phrases by Embedding the Dictionary"](http://arxiv.org/pdf/1504.00548v3.pdf).
* **wiki_idf.pkl**: a python dictionary containing 374,000 words where the values are the Inverse Document Frequencies (IDF) computed from the English Wikipedia.
* **cat_pages.pkl**: a python dictionary where the keys are the Wikipedia's categories and the values are the lists of pages inside the categories.
* **page_pos.pkl**: a python dictionary where the keys are the articles' titles and the values are the page positions (offset in bytes) in the Wikipedia's dump file.
* **page_size.pkl**: a python dictionary where the keys are the articles' titles and the values are the page sizes (in bytes) in the Wikipedia's dump file.


## Accessing the Dataset

Due to their large sizes, the wikipedia articles and queries files are stored in the HDF5 format,
which allows fast access without having to load them entirely into memory.

We provide wrapper classes (wiki.py and qp.py) to make your life easier when accessing these files.

For instance, the text and links of the "Machine Learning" article can be accessed using the python code below (the h5py package is required):

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


##Creating Your Own Dataset

If you want to generate your own dataset from a website, run 'create_dataset.py'. Don't forget to change the paths inside the parameters.py file to point to where you want to save the dataset.

The file 'wiki_parser.py' contains the code to parse an wikipedia article from the dump file, given its title. If you want to parse a specific website, you will have to create your own parser method. The only input to the parse() function is the url of the webpage and it must return the content (as a string) and a list of hyperlinks inside the page.

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

For the 4-hops, 4-sentences model, it takes around 30 hours to perform 100k minibatch updates on a K80 GPU.


##Dependencies

To run the code, you will need:
* Python 2.7
* [NumPy](http://www.numpy.org/)
* [scikit learn](http://scikit-learn.org/stable/index.html)
* [Theano](http://deeplearning.net/software/theano/)
* [NLTK](http://www.nltk.org/)
* [h5py](http://www.h5py.org/)

We recommend that you have at least 16GB of RAM. If you are going to use a GPU, the card must have at least 6GB.



## Reference

If you use this code as part of any published research, please acknowledge the
following paper:

**"TITLE"**  
AUTHORS*To appear CONFERENCE (2016)*

    @article{x,
        title={x},
        author={x},
        journal={x},
        year={}
    } 

## License

The code is released under a [revised (3-clause) BSD License](http://directory.fsf.org/wiki/License:BSD_3Clause).

