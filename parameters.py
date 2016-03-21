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
vocab_path = '../../data/D_cbow_pdw_8B.pkl' # Path to the python dictionary containing the vocabulary.
wordemb_path = '../../data/D_cbow_pdw_8B.pkl' # Path to the python dictionary containing the word embeddings.
idf_path = '../../data/wiki_idf.pkl' # Path to the IDF dictionary.
pages_path = '../../data/wiki.hdf5' # Path to save the articles and links.
pages_emb_path = '../../data/wiki_emb.hdf5' # Path to save articles embeddings (set to None to not compute it).
qp_path_pre = '../../data/queries_and_path.hdf5' # Path to save queries and paths.



#########################
# Wikipedia  parameters #
#########################
compute_page_pos = False # Compute or not the page positions in the Wikipedia dump file
dump_path = '../../data/enwiki-latest-pages-articles.xml' # Path to the wikipedia dump file.
page_pos_path = '../../data/page_pos.pkl' # Path to save the dictionary that stores each article position in the wikipedia dump file.
cat_pages_path = '../../data/cat_pages.pkl' # Path to save the dictionary that stores the pages in each wikipedia category.



####################
# Model parameters #
####################
qp_path = '../../data/queries_paths_4hops_4sentences.hdf5' # Path to load queries and paths.
#visited_pages_path = '/scratch/rfn216/QA_Allen/data/pred_pages.pkl' # Path to save the visited pages by the model.
visited_pages_path = None
dim_proj=500  # LSTM number of hidden units.
dim_emb=500  # word embedding dimension.
patience=100  # Number of epochs to wait before early stop if no progress.
max_epochs=5000  # The maximum number of epochs to run.
dispFreq=10  # Display to stdout the training progress every N updates.
lrate=0.0002  # Learning rate for sgd (not used for adadelta and rmsprop).
erate = 0.1  # multiplier for the entropy regularization. Only used when act_sel='softmax'.
saveto='model.npz'  # The best model will be saved there.
validFreq=10000  # Compute the validation error after this number of update.
saveFreq=10000  # Save the parameters after every saveFreq updates.
batch_size_train=16  # The batch size during training.
batch_size_pred=4  # The batch size during training.
max_hops_train = 2 # maximum number of pages to be visited before giving up - training.
max_hops_pred = 4 # maximum number of pages to be visited before giving up - prediction.
learning = 'supervised' # Valid options are: 'supervised', 'reinforce', and 'q_learning'.
act_sel = 'softmax' # Action selection types: 'epsilon-greedy' and 'softmax'. Only used when learning='q_learning', otherwise, act_sel='softmax'.
encoder='LSTM' # valid options are 'LSTM' or 'RNN'.
n_layers=1 # number of lstm layers.
#reload_model='./model.npz'  # Path to a saved model we want to start from.
reload_model=False  # Path to a saved model we want to start from.
idb=False # use input-dependent baseline. Should be used only when lerning='reinforce'.
train_size=1000  # If >0, we keep only this number of train examples when measuring accuracy.
valid_size=1000  # If >0, we keep only this number of valid examples when measuring accuracy.
test_size=1000  # If >0, we keep only this number of test examples when measuring accuracy.
outpath = "out.log" # where to save the logs file.
fixed_wemb = True # set to true if you don't want to learn the word embedding weights.
k = 4 # beam search width. Used in prediction only.
att_query = False # if True, use attention mechanism on queries.
att_doc = False # if True, use attention mechanism on documents.
att_segment_type = 'section' # Type of segment document for attention. Valid values are 'section' and 'sentence'. Only used when att_doc=True.
max_segs_doc = 10 # Maximum number of segments per document. Only used when att_doc=True.
att_window = 3 # attention window. Only used when att_query=True or att_doc=True.
mixer = 0 # decrease one hop of supervision after <mixer> iterations. Should be used only when lerning='reinforce'. Set to 0 to disable it.
replay_mem_size = 1000000 # Experience replay memory size. Only used when learning='q_learning'.
replay_start = 10000 # Start updating weights only after <replay_start> steps. Only used when learning='q_learning'.
freeze_mem = 10000 # Only update memory after this number, except when number of iterations < replay start. Only used when learning='q_learning'.
selective_mem = -1 # if >=0.0, this parameter corresponds to the percentage of memories with reward=1 to be stored. Only used when learning='q_learning'.
update_freq = 100 # Interval between each next model weights update. Only used when learning='q_learning'.
epsilon_start = 1.0 # Starting value for epsilon. Only used when learning='q_learning'.
epsilon_min = 0.05 # Minimum epsilon. Only used when learning='q_learning'.
epsilon_decay = 1 # Number of steps/updates to minimum epsilon. Only used when learning='q_learning'.
discount = 0.99 # Discount factor. Only used when learning='q_learning'.
clip = 1.0 # clip the cost at the this value. Only used when > 0 and learning='reinforce' or 'q_learning'.
load_emb_mem = True # If true, load the entire hdf5 file specified at <pages_emb_path> onto memory. This speeds up the code but the memory must be at least the size of the hdf5 file.


