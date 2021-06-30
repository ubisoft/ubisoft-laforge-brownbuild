import preprocessing.get_data as get_data
import preprocessing.sub_sets as sub_sets
import preprocessing.vectorization as vectorization
import classification.baseline as baseline
import classification.classification_XGboost as classification_XGBoost
import classification.metrics as metrics

import tools.pick_call as pick_call
import os
import time
import sys
import getopt
import ast

# PATH_experiment is the name of the folder that will contain the pickles
# of the experiments.
PATH_experiment = 'experiments/'
if not os.path.exists(PATH_experiment):
    os.mkdir(PATH_experiment)


class Experiment():
    '''
    Experiment class

    contains all the settings you want to define for you experiments:
    - path_data    : path to the build log dataset already processed (see go processor)
    - setting_name : setting identifier (will be the name of you pickle folder)
    - ngram        : list of N considered for the ngram feature_extraction
    - oversampling : if the training set must be oversampled or not
    - fail_mask    : mask to filter which subsets must only contain fails (Train/None/All)
    - kbest_thresh : number of features that need to be selected by kbest_t
    - alpha        : weight of model 1 in prediction (and 100-alpha is weight of model 2)
                     value in 0-100 (multiples of 10)
    - beta         : threshold for prediction flaky.
                     value in 10-90 (multiples of 10)
    '''

    def __init__(self,
                 path_data,
                 setting_name='default',
                 ngram=[2],
                 oversampling=True,
                 fail_mask='Train',
                 kbest_thresh=300,
                 alpha=70,
                 beta=10.
                 ):
        self.path_data = path_data
        self.path_exp = PATH_experiment + setting_name + '/'

        if not os.path.exists(self.path_exp):
            os.mkdir(self.path_exp)
        # Hyperparam
        self.ngram = ngram
        self.oversampling = oversampling
        self.fail_mask = fail_mask
        self.kbest_thresh = kbest_thresh
        self.alpha = alpha
        self.beta = beta


def results_print(BASELINES, XGB):
    want = ['f1', 'precision', 'recall', 'specificity']

    list = ['Run', 'F1-Score', 'Precision', 'Recall', 'Specificity']
    print('{:12s} | {:12s} {:12s} {:12s} {:12s} |'.format(*list))
    print('-' * 68)

    for BASE in BASELINES:
        list = [BASE.upper()] + [str(round(100*BASELINES[BASE][a], 1)) for a in want]
        print('{:12s} | {:12s} {:12s} {:12s} {:12s} |'.format(*list))

    list = ['XGB'] + [str(round(100*XGB[a], 1)) for a in want]
    print('{:12s} | {:12s} {:12s} {:12s} {:12s} |'.format(*list))


def run_cross_val(p, recompute=False):
    '''
    Cross validation run with experiment p.
    This function only trains one model with randomly selected Train(90%)/Valid(5%)/Test(5%) sets.

    The different stages of the run are pickled to reduce second run computation time.
    If you don't want to use the existing pickle, set recompute = True.
    '''
    start_time = time.time()

    DATA = pick_call.run_and_pickle(get_data.get_data,
                                    {'P': p},
                                    p.path_exp + 'data.p',
                                    recompute=recompute)
    SETS = pick_call.run_and_pickle(sub_sets.sub_sets,
                                    {'P': p, 'res': DATA},
                                    p.path_exp + 'sets.p',
                                    recompute=recompute)
    VECTORS = pick_call.run_and_pickle(vectorization.vectorization,
                                       {'P': p, 'sets': SETS},
                                       p.path_exp + 'vectors.p',
                                       recompute=recompute)

    BASELINES = baseline.baseline(p, DATA)
    BIG = classification_XGBoost.classify_XGBoost(p, VECTORS)
    interest = BIG['%.1fvar_%dtresh' % (float(p.beta), p.alpha)]['result']

    results_print(BASELINES, interest)
    print('===== TOTAL TIME: ', round(time.time() - start_time, 2), 'sec =====')


def run_10cross_val(p, recompute=False):
    '''
    double 10fold cross validation run with experiment p.
    This function does a 10fold cross validation with 2 runs at each fold (see paper).

    The different stages of the run are pickled to reduce second run computation time.
    If you don't want to use the existing pickle, set recompute = True.
    '''
    start_time = time.time()

    DATA = pick_call.run_and_pickle(get_data.get_data,
                                    {'P': p},
                                    p.path_exp + 'data.p',
                                    recompute=recompute)

    sets_10fold = pick_call.run_and_pickle(sub_sets.tenfolds_half_sets,
                                           {'res': DATA},
                                           p.path_exp + 'sets_10fold.p',
                                           recompute=recompute)

    all_PRED = {}
    for fold in range(10):
        for turn in range(2):
            SETS = sub_sets.sub_sets_10fold(
                **{'P': p, 'sets': sets_10fold, 'fold': fold, 'turn': turn})

            VECTORS = pick_call.run_and_pickle(vectorization.vectorization,
                                               {'P': p, 'sets': SETS},
                                               p.path_exp +
                                               'vectors_10fold_run%d_turn%d.p' % (fold+1, turn+1),
                                               recompute=recompute)

            BIG = classification_XGBoost.classify_XGBoost(p, VECTORS)

            for i in BIG:
                if i not in all_PRED:
                    all_PRED[i] = {
                        'real': list(
                            VECTORS['test']['y']),
                        'pred': BIG[i]['pred']}
                else:
                    all_PRED[i]['real'] += VECTORS['test']['y']
                    all_PRED[i]['pred'] += BIG[i]['pred']

    all_BIG = {
        i: metrics.compute_metrics(
            all_PRED[i]['real'],
            all_PRED[i]['pred']) for i in all_PRED}
    interest = all_BIG['%.1fvar_%dtresh' % (float(p.beta), p.alpha)]
    BASELINES = baseline.baseline(p, DATA)

    results_print(BASELINES, interest)

    print('===== TOTAL TIME: ', round(time.time() - start_time, 2), 'sec =====')


if __name__ == "__main__":
    try:
        opts, _ = getopt.getopt(sys.argv[1:], 'd:', ['path_data=',
                                                     'setting_name=',
                                                     'ngram=',
                                                     'oversampling=',
                                                     'fail_mask=',
                                                     'kbest_thresh=',
                                                     'alpha=',
                                                     'beta=',
                                                     '10fold',
                                                     'recompute'])
    except getopt.GetoptError:
        print('main.py -d <data_path> [--setting_name <string>] [--ngram <list int>] [--oversampling <bool>] [--fail_mask <Train/Valid/All>] [--kbest_thresh] <int>] [--alpha <int>] [--beta <int]')
        sys.exit(2)

    fun = run_cross_val
    recompute = False

    params = {}
    for arg, val in opts:
        if arg in ['-d', '--path_data']:
            params['path_data'] = val
        elif arg == '--setting_name':
            params['setting_name'] = val
        elif arg == '--ngram':
            assert ast.literal_eval(val) in [[1], [2], [1, 2]]
            params['ngram'] = ast.literal_eval(val)
        elif arg == '--oversampling':
            assert val in ['True', 'False']
            params['oversampling'] = ast.literal_eval(val)
        elif arg == '--fail_mask':
            assert val in ['Train', 'None', 'All']
            params['fail_mask'] = val
        elif arg == '--kbest_thresh':
            assert int(val) > 0
            params['kbest_thresh'] = int(val)
        elif arg == '--alpha':
            assert int(val) in [i*10 in range(,11)]
            params['alpha'] = int(val)
        elif arg == '--beta':
            assert int(val) in [i*10 in range(1,10)]
            params['beta'] = int(val)
        elif arg == '--10fold':
            fun = run_10cross_val
        elif arg == '--recompute':
            recompute = True

    print('Experiment:', params)
    p = Experiment(**params)

    fun(p, recompute)

    # python .\main.py -p 'D:/DATA_pickle/DATA_graphviz_pickle/' --ngram [1] --oversampling=True
