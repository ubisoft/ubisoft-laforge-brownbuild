import xgboost as xgb
import shap
import pandas as pd
import numpy as np
import classification.metrics as metrics


def pred_xgboost(sets):
    '''
    Trains a xgboost model on the sets (train/valid/test).

    Parameters:
    - sets: list of dictionaries with keys=train/valid/test and values=subsets.
    Outputs:
    - bst: xgboost model.
    - shap_val: list of shap values for each prediction.
    - pred: list prediction rounded to integer (1 for flaky, 0 for safe).
    - pred: list prediction value between 0.0 and 1.0.
    '''
    dtrain = xgb.DMatrix(sets['train']['X'], label=sets['train']['y'])
    dvalid = xgb.DMatrix(sets['valid']['X'], label=sets['valid']['y'])
    dtest = xgb.DMatrix(sets['test']['X'])

    param = {
        'max_depth': 100,
        'eta': 1,
        'silent': 1,
        'objective': 'binary:logistic'}
    evallist = [(dtrain, 'train'), (dvalid, 'valid')]

    bst = xgb.train(
        param,
        dtrain,
        50,
        evals=evallist,
        maximize=True,
        early_stopping_rounds=3,
        verbose_eval=0)
    explainer = shap.TreeExplainer(bst)

    shap_val = {}
    shap_val['train'] = explainer.shap_values(sets['train']['X'])
    shap_val['valid'] = explainer.shap_values(sets['valid']['X'])
    shap_val['test'] = explainer.shap_values(sets['test']['X'])

    pred = bst.predict(dtest)
    pred_prob = pred
    pred = [int(round(e)) for e in pred]

    return bst, shap_val, pred, pred_prob


def classify_XGBoost(P, sets):
    '''
    Trains our two layer classification XGBoost model.

    Parameters:
    - P   : Experiment object representing the current experiment set-up
    - sets: list of dictionaries with keys=train/valid/test and values=subsets.
    Outputs:
    - BIG : dictionary containing the predictions for all the alpha and beta values 
            considered in the paper.
            Key=code including alpha and beta ('beta'var_'alpha'tresh)
            Value=a dictionary containing the prediction and the result metrics.
    '''
    ### FIRST MODEL ###
    model1, shap_val, pred, pred_prob = pred_xgboost(sets)

    ### SECOND MODEL ###
    list_add = ["rerun", "commit_since_flaky"]

    select_col = np.std(shap_val['train'], axis=0) != 0
    select_col = [i for i, e in enumerate(select_col) if e]
    mat2 = np.concatenate((np.array(shap_val['train'][:, select_col]), np.array(
        [[e[k] for k in list_add] for e in sets["train"]["info"]])), axis=1)
    valid_X2 = np.concatenate((np.array(shap_val['valid'][:, select_col]), np.array(
        [[e[k] for k in list_add] for e in sets["valid"]["info"]])), axis=1)
    test_X2 = np.concatenate((np.array(shap_val['test'][:, select_col]), np.array(
        [[e[k] for k in list_add] for e in sets["test"]["info"]])), axis=1)

    second_sets = {}
    second_sets['train'] = {'X': mat2, 'y': sets["train"]["y"]}
    second_sets['valid'] = {'X': valid_X2, 'y': sets["valid"]["y"]}
    second_sets['test'] = {'X': test_X2, 'y': sets["test"]["y"]}
    X_train = pd.DataFrame(mat2)  # , columns=new_columns)
    X_valid = pd.DataFrame(valid_X2)  # , columns=new_columns)
    X_test = pd.DataFrame(test_X2)  # , columns=new_columns)

    model2, shap_val2, pred_2, pred_prob_2 = pred_xgboost(second_sets)

    BIG = {}
    for alpha in range(0, 110, 10):
        for beta in range(10, 100, 10):
            pred_prob_ranged = [(a * (100. - beta) + b * beta) /
                                100.0 for a, b in zip(pred_prob, pred_prob_2)]
            pred_ranged = [int(e >= float(alpha / 100.0))
                           for e in pred_prob_ranged]

            id = '%.1fvar_%dtresh' % (float(beta), alpha)
            result_ranged = metrics.compute_metrics(sets["test"]["y"], pred_ranged)

            BIG[id] = {"pred": pred_prob_ranged, "result": result_ranged}

    pred_prob_FINAL = [(a + b) / 2. for a, b in zip(pred_prob, pred_prob_2)]
    pred_FINAL = [int(round(e)) for e in pred_prob_FINAL]

    explainer = shap.TreeExplainer(model1)
    explainer2 = shap.TreeExplainer(model2)

    return BIG
