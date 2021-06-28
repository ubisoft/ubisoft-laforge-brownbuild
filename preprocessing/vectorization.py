from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2

import math
from scipy.sparse import csr_matrix

def set_to_corpus(sets, target=None):
    '''
    From the wordcount sets 'sets', generates a corpus of phrases only containing 
    words from 'target'.
    If target==None, then creates a corpus with all the words in the sets.

    Parameters: 
    - sets  : dictionary with keys=word and values=subsets.
    - target: list of words or None (default=None).
    Output:
    - corpus: string of words, respecting the word count in sets.

    The corpus is created by concatenating all the words in the wordcount set the 
    number of times given in wordcount.
    Ex: sets = [{'a':3, 'b':1}, {'a':2, 'c':2}]
        target = ['a', 'b']
        
        out = ['a a a b', 'a a']
    '''
    here_sets = sets['word_count'].tolist()

    if target is not None:
        corpus = [' '.join(
            [w for w in target if w in dic for i in range(dic[w])]) for dic in here_sets]
    else:
        corpus = [' '.join([w for w in dic for i in range(dic[w])])
                  for dic in here_sets]
    return corpus


def tf_idf(sets, target=None, only_train=False):
    '''
    Computes the tfidf metric for a dictionary of sets 'sets' where each value is a 
    wordcount set, and the keys are the subsets names ('train', 'valid', 'test'). 
    'target' is a list of words that define the vocabulary to use for
    the computation (mask the other words).  
    If 'only_train' is set to True, only consider the key='train' in the set. If 
    not, consider all keys.

    Parameters: 
    - sets      : list of dictionaries with keys=train/valid/test and 
                  values=subsets.
    - target    : list of words or None (default=None).
    - only_train: boolean (default=False)
    Outputs:
    - M         : dictionary of keys=train/valid/test (or just train) and values=tfidf 
                  matrix 
    - features  : list of words/features of the tfidf matrices (names of the columns)

    Returns the tfidf matrices for all the considered keys in the 'sets' as a 
    dictionary 'M' and a list of the feature names of the tfidf matrices.
    '''
    corpus = {}
    for who in sets:
        corpus[who] = set_to_corpus(sets[who], target=target)

    M = {}
    vectorizer = TfidfVectorizer()

    M['train'] = vectorizer.fit_transform(corpus['train'])
    if not only_train:
        M['valid'] = vectorizer.transform(corpus['valid'])
        M['test'] = vectorizer.transform(corpus['test'])
    
    features = vectorizer.get_feature_names()
    return M, features


def kbest(P, M, W, Y):
    '''
    Select the Kbest features for a given matrices 'M', with feature names 'W' and 
    labels 'Y'. The K number of features to select is given in the Experiment
    object 'P'.

    Parameters:
    - P: Experiment object representing the current experiment set-up
    - M: A matrix (size: set_size x nbr_features)
    - W: A list of features/words (size: nbr_features)
    - Y: A list of labels (size: set_size)
    Output:
    - k_selected: list of features selected (size: P.kbest_thresh)
    '''
    model = SelectKBest(chi2, k=P.kbest_thresh)
    X_new = model.fit_transform(M, Y)

    k_selected = [W[i] for i in model.get_support(True)]
    return k_selected


def X_values(P, sets):
    '''
    Computes the TF-IDF matrices, following the paper's iterative vectorization 
    approach.

    Parameters: 
    - P         : Experiment object representing the current experiment set-up
    - sets      : list of dictionaries with keys=train/valid/test and 
                  values=subsets.
    Outputs:
    - M_tfidf   : dictionary with keys=train/valid/test and values=tfidf matrix 
    - features  : list of words/features of the tfidf matrices (names of the columns)
    '''
    iter_size = 1000 #size of the sub training sets
    N = math.ceil(sets['train'].shape[0] / iter_size)
    k_selected = []
    for i in range(N): 
        # generate tfidf matrices + kbest selecting for each sub training set
        sub_set = {'train': sets['train'].iloc[i * iter_size:(i + 1) * iter_size]}
        M_tfidf, target = tf_idf(sub_set, only_train=True)
        Y_tfidf = y_values(P, sub_set)
        k_selected += kbest(P, M_tfidf['train'], target, Y_tfidf['train'])
        
    # final kbest selection on the union of the preselected features
    k_selected = list(set(k_selected))
    sub_set = {'train': sets['train']}
    M_tfidf, target = tf_idf(sub_set, target=k_selected, only_train=True)
    Y_tfidf = y_values(P, sub_set)
    final_k_selected = kbest(P, M_tfidf['train'], target, Y_tfidf['train'])

    # final tfidf matrices
    M_tfidf, target = tf_idf(sets, target=final_k_selected)

    return M_tfidf, target


def y_values(P, sets):
    '''
    Computes the Y/label vectors were 1=flaky and 0=safe.

    Parameters: 
    - P   : Experiment object representing the current experiment set-up
    - sets: list of dictionaries with keys=train/valid/test and 
            values=subsets.
    Outputs:
    - Y    : dictionary with keys=train/valid/test and values=label vector
    '''
    Y = {}
    for who in sets:
        Y[who] = [int(e == "flaky") for e in sets[who]["flaky"].tolist()]
    return Y


def vectorization(P, sets):
    ''' 
    Vectorizes the subsets in 'sets' using the tfidf and kbest selection.

    Parameters: 
    - P      : Experiment object representing the current experiment set-up
    - sets   : list of dictionaries with keys=train/valid/test and 
               values=subsets.
    Outputs:
    - VECTORS: dictionary with keys=train/valid/test and values=info_dictionary.
               info_dictionary are dictionary with keys:
                    - X: the tfidf matrix
                    - y: the label vector
                    - info: list of dictionary with additional metrics (see paper)
                    - feat: list of features (the column names of the tfidf matrix)
    '''
    M, target = X_values(P, sets)
    Y = y_values(P, sets)

    VECTORS = {}
    for who in M:
        VECTORS[who] = {
            'X': M[who],
            'y': Y[who],
            'info': sets[who]['info'],
            'feat': target}
    return VECTORS
