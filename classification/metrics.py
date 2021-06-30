import sklearn
import sklearn.metrics as metrics
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def compute_metrics(y, pred):
    '''
    Computes the performance metrics (precision/recallspecificity/f1) on the 
    prediction regarding the actual label.

    Parameters: 
    - y     : list of true labels (1 for flaky, 0 for safe).
    - pred  : list of prediction value between 0.0 and 1.0.
    Output:
    - result: dictionnary containing all the performance metrics. 
              Keys= name of the metrics and value= the value of the metric.
    '''
    pred_round = np.array([(int(round(i))) for i in pred])

    result = {}
    result['accuracy'] = sklearn.metrics.accuracy_score(y, pred_round)
    result['precision'] = sklearn.metrics.precision_score(y, pred_round)
    result['recall'] = sklearn.metrics.recall_score(y, pred_round)
    result['f1'] = sklearn.metrics.f1_score(y, pred_round)
    result['specificity'] = sklearn.metrics.recall_score(
        [1 - e for e in y], [1 - e for e in pred_round])
    return result
