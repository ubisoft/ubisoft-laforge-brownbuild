from os import listdir, path
import numpy as np
import re
import pandas as pd
from datetime import datetime

MAX_NGRAM = 2
file_regex = r"((.*_.*_.*_.*_.*_.*)_(.*)_(.*)_([01])(_(.*))?)-processed\.csv"
date_regex = "%Y_%m_%d_%H_%M_%S"


def get_text_count(file):
    '''
    Get the word count in the file with filename 'file'.
    The function returns a list of dictionary of word count for words generated
    with ngram where N in 1..MAX_NGRAM.
    '''
    with open(file) as f:
        txt = f.read()
        sep_txt = [e for e in txt.split('#') if e != ""]
        sep_txt = sep_txt[:MAX_NGRAM]

        dic = [{} for e in range(MAX_NGRAM)]
        count = 0
        for e in sep_txt:
            loc = {}
            for line in e.split('\n'):
                row = line.split(',')
                if len(row) == 2 and len(row[0]) > 2:
                    loc[row[0]] = int(row[1])
            dic[count] = loc
            count += 1
    return dic


def get_log_data(file, DATA_PATH):
    '''
    Returns a list representation of the job given in the file with filename 
    'file' at the path 'DATA_PATH'.
    '''
    m = re.match(file_regex, file)
    if(m):
        date = datetime.strptime(m.group(2), date_regex)
        jobID = m.group(3)
        commitID = m.group(4)
        status = int(m.group(5))
        jobName = m.group(7)
        filename = DATA_PATH + file
        word_count = get_text_count(filename)
        loc = [date, jobID, commitID, status, jobName, filename] + word_count

        return loc
    else:
        return "ERROR"


def flaky_state(mean):
    '''
    Returns if a job is flaky or safe.

    Parameters:
    - mean: float. Mean result of the jobs run.
    '''
    if 0 < mean < 1:  # unsteady results => flaky
        return "flaky"
    return "safe"


def flaky_state_all(res):
    '''
    Computes the flaky column.

    Parameters:
    - res   : dataset in a pandas dataframe format.
    Output:
    - res   : modified dataset in a pandas dataframe format, with added 
              'flaky' column.
    '''
    list_aggr = ["commitID", "jobName"]
    COMJOB_mean_status = res.groupby(list_aggr)["status"].mean()
    COMJOB_flaky_state = [flaky_state(a) for a in COMJOB_mean_status.tolist()]

    indexes = [a for a in COMJOB_mean_status.index]

    all_flaky_state_res = [COMJOB_flaky_state[indexes.index((a, b))] for a, b in zip(
        res["commitID"].tolist(), res["jobName"].tolist())]

    all_flaky_state = pd.DataFrame(all_flaky_state_res, columns=["flaky"])
    all_flaky_state["flaky"] = all_flaky_state_res

    res["flaky"] = all_flaky_state["flaky"]
    return res


def get_data(P):
    '''
    Gets data for Experiment object 'P'.

    Parameters:
    - P  : Experiment object representing the current experiment set-up
    Output:
    - res: dataset in a pandas dataframe format.
    '''
    res = []

    list_log = sorted(listdir(P.path_data))

    res = np.array([get_log_data(f, P.path_data)
                   for f in list_log if re.match(file_regex, f)])
    colnames = ["date", "jobID", "commitID", "status", "jobName", "filename"] + \
        ["word_count_ngram_" + str(i) for i in range(1, 1 + MAX_NGRAM)]

    res = pd.DataFrame(res, columns=colnames)
    res["status"] = res["status"].astype('int')

    res = flaky_state_all(res)

    return res.reset_index(drop=True)
