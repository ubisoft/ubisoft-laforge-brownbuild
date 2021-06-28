import math
from random import shuffle
import pandas as pd


def shuffle_df(df):
    ''' 
    Shuffles lines in a pandas dataframe.

    Parameter:
    - df : a pandas dataframe.
    Output:
    - shuffled: a shuffled pandas dataframe.
    '''
    index = [e for e in range(df.shape[0])]
    shuffle(index)
    
    shuffled = df.iloc[index]
    return shuffled

def get_word_count(res, ngrams):
    '''
    Computes the word_count column depending on the ngram considered.

    Parameters:
    - res   : dataset in a pandas dataframe format.
    - ngrams: list of the N values considered.
    Output:
    - res   : modified dataset in a pandas dataframe format, with added 
              'word_count' column.
    '''
    if len(ngrams) == 1:
        res["word_count"] = res["word_count_ngram_" + str(ngrams[0])]
        return res

    loc = [{} for i in range(res.shape[0])]
    for i in ngrams:
        loc = [a.update(b) for a, b in zip(
            loc, res["word_count_ngram_" + str(i)].tolist())]

    res["word_count"] = loc
    return res


def get_since_flaky(res):
    '''
    Computes the commit_since_flaky metric.

    Parameters:
    - res: dataset in a pandas dataframe format.
    Output:
    - l  : vector of size res_nbr_row containing the commit_since_flaky metric.
    '''
    commit_since_flaky = res.groupby(["commitID"]).agg({'flaky': lambda x: sum(
        x == "flaky") > 0, 'date': lambda x: min(x), 'commitID': lambda x: list(x.index)})
    commit_since_flaky.sort_values(by='date')

    l = []
    count = 0
    for e in commit_since_flaky["flaky"].tolist():
        l += [count]
        count += 1
        if e:
            count = 0
    commit_since_flaky["since_flaky"] = l

    l = [0] * (res.shape[0])
    for i in range(commit_since_flaky.shape[0]):
        for e in commit_since_flaky["commitID"].tolist()[i]:
            l[e] = commit_since_flaky["since_flaky"].tolist()[i]

    return l


def get_info_rerun(res):
    '''
    Computes the 'info' column depending, which contains the other metrics 
    (#rerun, #fail, #success, commit_since_flaky). See paper.

    Parameters:
    - res   : dataset in a pandas dataframe format.
    Output:
    - res   : modified dataset in a pandas dataframe format, with added 
              'info' column.
    '''    
    info = [{} for e in range(res.shape[0])]

    curr_commitXjob = ""
    curr_dic = {}
    list = []
    max = 0

    since_flaky = get_since_flaky(res)
    for i, row in res.sort_values(
            by=["commitID", "jobName", "date"]).iterrows():

        if curr_commitXjob != row["commitID"] + row["jobName"]:
            curr_commitXjob = row["commitID"] + row["jobName"]

            curr_dic = {
                "rerun": 0,
                "fail": 0,
                "success": 0,
                "commit_since_flaky": 0}

        curr_dic["commit_since_flaky"] = since_flaky[i]
        info[i] = curr_dic.copy()
        
        curr_dic["rerun"] += 1
        curr_dic["fail"] += row["status"]
        curr_dic["success"] += 1 - row["status"]

    res["info"] = info

    res["commit_since_flaky"] = since_flaky
    return res


def just_failure(res):
    '''
    Masks the non-failure jobs form a set 'res'.

    Parameters:
    - res: full dataset in a pandas dataframe format.
    Output:
    - res: modified dataset in a pandas dataframe format, containing only failing 
           jobs.
    '''        
    res = res.iloc[[i for i, e in enumerate(res["status"].tolist()) if e == 1]]
    return res


def mask_failure(sets, cond):
    '''
    Masks the non-failure jobs form a set 'res'.

    Parameters:
    - sets: list of dictionaries with keys=train/valid/test and values=subsets.
    - cond: mask condition. 
            If cond=='Train', mask non-failure jobs in the training and test set.
            If cond=='All'  , mask non-failure jobs in all sets.
            Else (cond=='None'), mask non-failure jobs in the test set.
    Output:
    - sets: list of dictionaries with keys=train/valid/test and values=modified 
            subsets according to the mask.
    ''' 
    mask = ['test'] #if cond == 'None'
    if cond == "Train":
        mask = ['train', 'test']
    if cond == "All":
        mask = ['train', 'valid', 'test']
    
    for who in mask:
        sets[who] = just_failure(sets[who])
    return sets


def oversampling(sets):
    '''
    Oversamples the sets to have the same ratio of failuresXflakiness in the sets.

    Parameters:
    - sets    : subset of dataframe format.
    Output:
    - new_sets: oversampled subset.
    ''' 
    sets = sets.reset_index(drop=True)
    ids_stat_flak = [sets[[a == stat and b == flak for a, b in zip(sets['status'], sets['flaky'])]].index.tolist(
    ) for stat in set(sets['status']) for flak in set(sets['flaky'])]
    max_len = max([len(e) for e in ids_stat_flak])

    new_ids = []
    for ids in ids_stat_flak:
        shuffle(ids)
        ids = ids * math.ceil(max_len / len(ids))
        new_ids += ids[:max_len]
    new_ids = sorted(new_ids)

    new_sets = sets.iloc[new_ids].reset_index(drop=True)
    return new_sets



### For random cross validation with train(90%)/valid(5%)/test(5%) ###

def random_sets_by_type(res, ids, flaky, want):
    '''
    Generates subsets for train(90%)/valid(5%)/test(5%). The subsets selected 
    jobs by commitID (all the jobs of a commitID will be in the same set). 
    If want=True, the subsets generated will only contain jobs from commitID that 
    have at least one flaky job. If want=False,  the subsets generated will only 
    contain jobs from commitID that only has safe builds.

    Parameters:
    - res  : full dataset in a pandas dataframe format.
    - ids  : list of list of ids, each sublist lists the line ids of jobs by 
             commitID. (size: nbr_commitID).
    - flaky: list of boolean, indicating if the commitID (related to the 'ids') 
             contains a flaky job or not.
    - want : boolean. If True, will only select commitIDs that contain at least a 
             flaky job. If False, will only select commitIDs that contain only safe
             builds.
    Output: 
    - sets : list of dictionaries with keys=train/valid/test and values=subsets.
    '''
    id_wanted = [i for i, e in zip(ids, flaky) if e == want]
    shuffle(id_wanted)
    perc10_lim = round(len(id_wanted) * 0.05)
    
    test = [e for l in id_wanted[:perc10_lim] for e in l]
    valid = [e for l in id_wanted[perc10_lim:(2 * perc10_lim)] for e in l]
    train = [e for l in id_wanted[(2 * perc10_lim):] for e in l]
    
    sets = {
        'train': res.iloc[train],
        'valid': res.iloc[valid],
        'test': res.iloc[test]}
    return sets

def random_sets(res):
    '''
    Generates subsets for train(90%)/valid(5%)/test(5%). The subsets selected 
    jobs by commitID (all the jobs of a commitID will be in the same set). The ratio 
    of commitID with flakiness and without is respected in the subsets.
    Parameters:
    - res  : full dataset in a pandas dataframe format.
    Output: 
    - sets : list of dictionaries with keys=train/valid/test and values=subsets.
    '''
    res = res.reset_index()
    commit_ids = res.groupby(
        ["commitID"]).apply(
        lambda x: list(
            x.index)).tolist()
    commit_flaky = res.groupby(
        ["commitID"])["flaky"].apply(
        lambda x: "flaky" in list(x)).tolist()

    flakys = random_sets_by_type(res, commit_ids, commit_flaky, True)
    safes  = random_sets_by_type(res, commit_ids, commit_flaky, False)

    data = {}
    for who in flakys:
        data[who] = shuffle_df(flakys[who].append(safes[who])).reset_index()

    return data

def sub_sets(P, res):
    '''
    Generates subsets for train(90%)/valid(5%)/test(5%), adds the necessary 
    columns (word_count and info), applies the mask failure and oversampling 
    as indicated in the Experiment object 'P'. 

    Parameters:
    - P   : Experiment object representing the current experiment set-up.
    - res : full dataset in a pandas dataframe format.
    Output: 
    - SETS: list of dictionaries with keys=train/valid/test and values=subsets.
    '''
    SETS = random_sets(res)

    for who in SETS:
        SETS[who] = get_word_count(SETS[who], P.ngram)
        SETS[who] = get_info_rerun(SETS[who])
    SETS = mask_failure(SETS, P.fail_mask)

    if P.oversampling:
        SETS['train'] = oversampling(SETS['train'])

    return SETS

### For random 10fold cross validation with train(90%)/valid(5%)/test(5%) ###

def tenfolds_half_by_type(res, ids, flaky, want):
    '''
    Generates 20 subsets of size 5%. The subsets selected 
    jobs by commitID (all the jobs of a commitID will be in the same subset). 
    If want=True, the subsets generated will only contain jobs from commitID that 
    have at least one flaky job. If want=False,  the subsets generated will only 
    contain jobs from commitID that only has safe builds.
    See the paper for the 10fold with two runs approach.

    Parameters:
    - res  : full dataset in a pandas dataframe format.
    - ids  : list of list of ids, each sublist lists the line ids of jobs by 
             commitID. (size: nbr_commitID).
    - flaky: list of boolean, indicating if the commitID (related to the 'ids') 
             contains a flaky job or not.
    - want : boolean. If True, will only select commitIDs that contain at least a 
             flaky job. If False, will only select commitIDs that contain only safe
             builds.
    Output: 
    - sets : list of 10 lists, where each sublist is composed of two subsets of 
             size 5%.
    '''
    id_wanted = [i for i, e in zip(ids, flaky) if e == want]
    shuffle(id_wanted)
    perc10_lim = round(len(id_wanted) * 0.05)

    sets = []
    
    for i in range(10):
        sets.append([res.iloc[[e for l in id_wanted[perc10_lim * (2 * i):perc10_lim * (2 * i + 1)] for e in l]],
                     res.iloc[[e for l in id_wanted[perc10_lim * (2 * i + 1):perc10_lim * (2 * i + 2)] for e in l]]])

    return sets


def tenfolds_half_sets(res):
    '''
    Generates 20 subsets of size 5%. The subsets selected jobs by commitID 
    (all the jobs of a commitID will be in the same set). The ratio 
    of commitID with flakiness and without is respected in the subsets.
    Parameters:
    - res  : full dataset in a pandas dataframe format.
    Output: 
    - sets : list of dictionaries with keys=0-9 and values=list of two subsets.
    '''
    res = res.reset_index()

    commit_ids = res.groupby(
        ["commitID"]).apply(
        lambda x: list(
            x.index)).tolist()
    commit_flaky = res.groupby(
        ["commitID"])["flaky"].apply(
        lambda x: "flaky" in list(x)).tolist()

    flakys = tenfolds_half_by_type(res, commit_ids, commit_flaky, True)
    safes = tenfolds_half_by_type(res, commit_ids, commit_flaky, False)

    data = {}
    for i, _ in enumerate(flakys):
        for j in range(2):
            data[i] = [
                shuffle_df(
                    flakys[i][0].append(
                        safes[i][0])).reset_index()]
            data[i] += [shuffle_df(flakys[i]
                                   [1].append(safes[i][1])).reset_index()]
    return data

def sub_sets_10fold(P, sets, fold=0, turn=0):
    '''
    Generates subsets for train(90%)/valid(5%)/test(5%), adds the necessary 
    columns (word_count and info), applies the mask failure and oversampling 
    as indicated in the Experiment object 'P', for the appropriate 'fold' and 'turn' 
    in 10fold cross validation. 

    Parameters:
    - P       : Experiment object representing the current experiment set-up
    - sets    : list of dictionaries with keys=0-9 and values=list of two subsets.
    - fold    : int. Value between 0-9 defining the fold we are currently running.
                For fold=X, the X element in sets is used for valid/test sets and 
                all the other are used for the training set.
    - turn    : int. Value between 0-1 defining the turn we are currently running.
                For fold=X and turn=Y, the element sets[X][Y] is used for the 
                validation set and the element sets[X][1-Y] is used for the test 
                set, and all the other are used for the training set. 
    Output: 
    - new_sets: list of dictionaries with keys=train/valid/test and values=subsets.
    '''
    new_sets = {}
    new_sets['train'] = pd.concat(
        [f for i in sets if i != fold for f in sets[i]], ignore_index=True)
    new_sets['valid'] = pd.concat([sets[i][turn]
                                  for i in sets if i == fold], ignore_index=True)
    new_sets['test'] = pd.concat([sets[i][1 - turn]
                                 for i in sets if i == fold], ignore_index=True)

    for who in new_sets:
        new_sets[who] = get_word_count(new_sets[who], P.ngram)
        new_sets[who] = get_info_rerun(new_sets[who])
    new_sets = mask_failure(new_sets, P.fail_mask)

    if P.oversampling:
        new_sets['train'] = oversampling(new_sets['train'])

    return new_sets
