import numpy as np


def baseline_metrics(flaky_rate, which):
    '''
    Computes a baseline 'which' performance metrics 
    (precision/recallspecificity/f1). 
    See paper for each baseline's description.

    Parameters:
    - flaky_rate: flaky rate of failing jobs in the project's dataset.
    - which     : name of the baseline (default=alwaysBrown).
    Output:
    - met       : dictionnary containing all the performance metrics. 
                  Keys= name of the metrics and value= the value of the metric.
    '''
    met = {}
    if which == 'random50':
        met['precision'] = flaky_rate
        met['recall'] = .5
        met['specificity'] = .5
        met['f1'] = flaky_rate / (1 + 2 * flaky_rate)
    elif which == 'randomB':
        met['precision'] = flaky_rate
        met['recall'] = flaky_rate
        met['specificity'] = 1 - flaky_rate
        met['f1'] = flaky_rate / 2
    else:  # which == 'alwaysBrown':
        met['precision'] = flaky_rate
        met['recall'] = 1
        met['specificity'] = 0
        met['f1'] = flaky_rate / (1 + flaky_rate)
    return met
    
def baseline(P, data):
    '''
    Computes all baselines performance metrics 
    (precision/recalléspecificity/f1). 
    See paper for each baseline's description.

    Parameters:
    - P       : Experiment object representing the current experiment set-up.
    - data    : full dataset in a pandas dataframe format.
    Output:
    - base_met: dictionnary with keys=name of baseline and value=metrics of 
                baseline. The metrics of a baseline is a dictionary with 
                keys= name of the metric and value= the value of the metric.
    '''
    fails = data[data['status'] == 1]['flaky'].tolist()
    brown = np.sum([e == 'flaky' for e in fails])
    true_fail = np.sum([e != 'flaky' for e in fails])

    flaky_rate = brown / (brown + true_fail)

    base_met = {}
    for which in ['random50', 'randomB', 'alwaysBrown']:
        base_met[which] = baseline_metrics(flaky_rate, which)
    return base_met


