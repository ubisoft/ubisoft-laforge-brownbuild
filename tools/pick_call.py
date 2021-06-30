import pickle
import time
import os


def pickle_dump(data, pick_file):
    with open(pick_file, "wb") as f:
        pickle.dump(data, f)


def pickle_load(pick_file):
    with open(pick_file, "rb") as f:
        data = pickle.load(f)
    return data


def run_and_pickle(fun, args, filename, recompute=False, do_print=True):
    start_time = time.time()
    print('Load ', filename, end=' ... ')
    if not recompute and os.path.exists(filename):
        COMPUTED = pickle_load(filename)
    else:
        print('(computing)', end=' ...')
        COMPUTED = fun(**args)
        pickle_dump(COMPUTED, filename)
    print('Done in', round(time.time() - start_time, 2), 'sec')
    return COMPUTED
