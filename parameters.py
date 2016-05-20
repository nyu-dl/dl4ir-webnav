######################
# Dataset parameters #
######################
n_samples = [[1e8,1000,1000],[1e8,20000,20000],[1e8,20000,20000]] # maximum number of samples for the training, validation and test sets per hop
max_hops = [2,4,8] # Maximum number of hops to be visited to extract queries.
max_hops_pages = 10 # Maximum number of hops.
max_sents = 5 # Maximum number of query sentences to be extracted in each page.
max_links = 300 # Maximum number of links a page can have. If it has more links than max_links, it is not included in the dataset. Set it to None if you want all pages to be included in the dataset.
min_words_query = 10 # Minimum number of words a query can have.
max_words_query = 30 # Maximum number of words a query can have.
n_words = 374000 # words for the vocabulary
n_consec = 4 # maximum number of consecutive sentences to form a query
root_page = 'category:main topic classifications'
vocab_path = './data/D_cbow_pdw_8B.pkl' # Path to the python dictionary containing the vocabulary.
wordemb_path = './data/D_cbow_pdw_8B.pkl' # Path to the python dictionary containing the word embeddings.
idf_path = './data/wiki_idf.pkl' # Path to the IDF dictionary.
pages_path = './data/wiki.hdf5' # Path to save the articles and links.
pages_emb_path = './data/wiki_emb.hdf5' # Path to save articles embeddings (set to None to not compute it).
pages_idx_path = './data/wiki_idx.hdf5' # Path to save articles' words as vocabulary indexes (set to None to not compute it).
qp_path_pre = './data/qp_345_24810.hdf5' # Path to save queries and paths.


#########################
# Wikipedia  parameters #
#########################
compute_page_pos = True # Compute or not the page positions in the Wikipedia dump file
dump_path = './data/enwiki-latest-pages-articles.xml' # Path to the wikipedia dump file.
page_pos_path = './data/page_pos_341.pkl' # Path to save the dictionary that stores each article position in the wikipedia dump file.
cat_pages_path = './data/cat_pages_341.pkl' # Path to save the dictionary that stores the pages in each wikipedia category.



###############################
# Jeopardy dataset parameters #
###############################
jeopardy_path = './data/JEOPARDY.csv' # Path to the csv containing jeopardy questions and answers.
jeopardy_n_samples = [1e8,10000,10000] # maximum number of samples for the training, validation and test sets.
max_candidates = 400 # maximum number of candidate documents that will be returned by the search engine.
search_engine = None # search engine to find candidate documents. Valid options are 'simple', 'lucene', 'google', and None.
create_index = False # If True, create index from wikipedia. If False, use the current one. Only used when search_engine='lucene'.
index_folder = './data/lucene_index/' # folder to store lucene's index. It will be created in case it does not exist.
 

###################################
# WebQuestions dataset parameters #
###################################
webquestions_train_path = './data/webquestions_train.json' # Path to the json file containing the WebQuestion training dataset.
webquestions_test_path = './data/webquestions_test.json' # Path to the json file containing the WebQuestion test 
max_candidates = 400 # maximum number of candidate documents that will be returned by the search engine.
search_engine = None # search engine to find candidate documents. Valid options are 'simple', 'lucene', 'google', and None.
create_index = False # If True, create index from wikipedia. If False, use the current one. Only used when search_engine='lucene'.
index_folder = './data/lucene_index/' # folder to store lucene's index. It will be created in case it does not exist.


####################
# Model parameters #
####################
qp_path = 'queries_paths_jeopardy.hdf5' # Path to load queries and paths.
dim_proj=2048  # LSTM number of hidden units.
dim_emb=500  # word embedding dimension.
patience=1000  # Number of epochs to wait before early stop if no progress.
max_epochs=5000  # The maximum number of epochs to run.
dispFreq=10  # Display to stdout the training progress every N updates.
lrate=0.0002  # Learning rate for sgd (not used for adadelta and rmsprop).
erate=0.1  # multiplier for the entropy regularization. Only used when act_sel='softmax'.
saveto='model.npz'  # The best model will be saved there.
validFreq=20000  # Compute the validation error after this number of updates.
saveFreq=20000  # Save the parameters after every saveFreq updates.
batch_size_train=16  # The batch size during training.
batch_size_pred=4  # The batch size during training.
max_hops_train = 10 # maximum number of pages to be visited before giving up - training.
max_hops_pred = 10 # maximum number of pages to be visited before giving up - prediction.
learning = 'supervised' # Valid options are: 'supervised', 'reinforce', and 'q-learning'.
act_sel = 'softmax' # Action selection types: 'epsilon-greedy' and 'softmax'. Only used when learning='q-learning', otherwise, act_sel='softmax'.
encoder='LSTM' # valid options are 'LSTM' or 'RNN'.
n_rnn_layers=8 # number of recurrent layers. must be >= 1.
n_doc_layers_nav=1 # number of layers after the document embedding in the navigation loop. must be >= 1.
n_doc_layers_final=1 # number of layers after the document embedding before the final scoring. must be >= 1.
scoring_layers_nav=[100,30] # list containing the number of hidden units in each intermediate scoring layer of the navigation loop. Set to [] if you want only one scoring layer.
scoring_layers_final=[100,30] # list containing the number of hidden units in each intermediate final scoring layer. Set to [] if you want only one scoring layer.
lambda_scoring = 0. # multiplication factor for the cost of the scoring layers. Set it to zero to disable scoring computation.
reward_type = None # Possible values are 'continuous', 'discrete', or None (in this case, no reward is computed. Used to speed up computations when learning='supervised'.
reload_model='model_d.npz'  # Path to a saved model we want to start from.
#reload_model=False  # Path to a saved model we want to start from.
idb=False # use input-dependent baseline. Should be used only when lerning='reinforce'.
train_size=5000  # If >0, we keep only this number of train examples when measuring accuracy.
valid_size=5000  # If >0, we keep only this number of valid examples when measuring accuracy.
test_size=5000  # If >0, we keep only this number of test examples when measuring accuracy.
fixed_wemb = True # set to true if you don't want to learn the word embedding weights.
k = 1 # beam search width. Used in prediction only.
dropout = 0.5 # If >0, <dropout> fraction of the units in the non-recurrent layers will be set to zero at training time.
att_query = True # if True, use attention mechanism on queries.
att_doc = False # if True, use attention mechanism on documents.
att_segment_type = 'section' # Type of segment document for attention. Valid values are 'section', 'subsection' and 'sentence'. Only used when att_doc=True.
max_segs_doc = 10 # Maximum number of segments per document. Only used when att_doc=True.
att_window = 3 # attention window. Only used when att_query=True or att_doc=True.
max_words = 100 # Maximum number of words per section (when att_segment_type = 'section') or sentence (when att_segment_type = 'sentence').
mixer = 0 # decrease one hop of supervision after <mixer> iterations. Should be used only when lerning='reinforce'. Set to 0 to disable it.
replay_mem_size = 1000000 # Experience replay memory size. Only used when learning='q-learning'.
replay_start = 50000 # Start updating weights only after <replay_start> steps. Only used when learning='q-learning'.
freeze_mem = 50000 # Only update memory after this number, except when number of iterations < replay start. Only used when learning='q-learning'.
prioritized_sweeping = 0.5 # if >=0.0, this parameter corresponds to the percentage of memories with reward=1 to be stored. Only used when learning='q-learning'.
saveto_mem = 'mem.pkl'  # The experience replay memory will be saved there. Only used when learning='q-learning'.
reload_mem = False  # Path to the pickle file containing the experience replay memory. If reload_mem==False, the replay memory will start empty. Only used when learning='q-learning'.
update_freq = -1 # Interval between each next model weights update. If < 2, the same network is used for computing
                 # the target Q-values, but the gradients are disconnected. Only used when learning='q-learning'.
epsilon_start = 1.0 # Starting value for epsilon. Only used when learning='q-learning'.
epsilon_min = 0.1 # Minimum epsilon. Only used when learning='q-learning'.
epsilon_decay = 1 # Number of steps/updates to minimum epsilon. Only used when learning='q-learning'.
discount = 0.99 # Discount factor. Only used when learning='q-learning'.
clip = 1.0 # clip the cost at the this value. Only used when > 0 and learning='reinforce' or 'q-learning'.
aug = 1 # Augmentation on training dataset.
path_thes_idx = "./data/th_en_US_new.idx" # Thesaurus
path_thes_dat = "./data/th_en_US_new.dat" # Thesaurus
load_emb_mem = True # If true, load the entire hdf5 file specified at <pages_emb_path> onto memory. This speeds up the code but the memory must be at least the size of the hdf5 file.
compute_emb = False # If True, compute word embeddings on the fly. If False, use precomputed word embeddings from <pages_emb_path>.
init_zero = True # If True, hidden states are initialized with zeros. If False, they are initialized with query embeddings.
weight_decay = 0.0 # weight decay multiplication factor.

