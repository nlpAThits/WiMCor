import os
import pickle

def dump_to_pickle(file, data):
    if not os.path.exists(file):
        open(file, 'x')
    with open(file, 'wb') as fp:
        pickle.dump(data, fp)

def load_from_pickle(file):
    with open(file, 'rb') as fp:
        data = pickle.load(fp)
    return data
