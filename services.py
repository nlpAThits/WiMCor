import matplotlib.pyplot as plt
import logging
import time
import os
import sys
from bs4 import BeautifulSoup
import numpy as np
import json
try:
   import cPickle as pickle
except:
   import pickle

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from seqeval.metrics import classification_report as seqlab_classification_report
from seqeval.metrics import accuracy_score as seqlab_accuracy_score
from seqeval.metrics import precision_score as seqlab_precision_score
from seqeval.metrics import recall_score as seqlab_recall_score
from seqeval.metrics import f1_score as seqlab_f1_score


def get_topk(arr, k, order):
    '''
        Get top-k elements from the array arr.
        Either top maximum or top minimum elements
    '''
    if order=='max':
        topk_indices = np.argsort(arr)[-k:][::-1]
    elif order=='min':
        topk_indices = np.argsort(arr)[:k]

    return arr[topk_indices]

def make_one_hot(arr):
    '''
        Accept a NumPy array
        and convert the array into one-hot format.
    '''
    one_hot_arr =  np.zeros((arr.size, arr.max()+1))
    one_hot_arr[np.arange(arr.size),arr] = 1
    return one_hot_arr

def plot_learning_curve(train_sizes, train_scores, test_scores, title='Learning Curve', fname=None, alpha=0.1):
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    plt.plot(train_sizes, train_mean, label='train score', color='blue', marker='o')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, color='blue', alpha=alpha)

    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.plot(train_sizes, test_mean, label='test score', color='red', marker='o')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, color='red', alpha=alpha)

    plt.title(title)
    plt.xlabel('Training Size')
    plt.ylabel('Performance Metric')
    plt.grid(ls='--')
    plt.legend(loc='best')

    if fname:
        plt.savefig(fname)
        print('Plot saved to file {}'.format(fname))

    plt.show()

def compute_mean_rpf(test_scores):
    test_scores = np.array(test_scores)

    acc = np.array(test_scores)[:, 0]
    pre = np.array(test_scores)[:, 1]
    rec = np.array(test_scores)[:, 2]
    f1s = np.array(test_scores)[:, 3]
    assert (len(acc) == len(pre) == len(rec) == len(f1s))

    print('--------------------------------------------')
    print('      Acc MEAN:{:6.3f} STD:{:6.3f} over {} runs(s)'.format(acc.mean(), acc.std(), len(acc)))
    print('Precision MEAN:{:6.3f} STD:{:6.3f} over {} runs(s)'.format(pre.mean(), pre.std(), len(pre)))
    print('   Recall MEAN:{:6.3f} STD:{:6.3f} over {} runs(s)'.format(rec.mean(), rec.std(), len(rec)))
    print(' F1-Score MEAN:{:6.3f} STD:{:6.3f} over {} runs(s)'.format(f1s.mean(), f1s.std(), len(f1s)))
    print('--------------------------------------------')

def compute_seqlab_metrics(y_true, y_pred):
    accuracy = seqlab_accuracy_score(y_true, y_pred)
    precision = seqlab_precision_score(y_true, y_pred)
    recall = seqlab_recall_score(y_true, y_pred)
    fscore = seqlab_f1_score(y_true, y_pred)
    print('Test Metrics: A = {:0.3f} P = {:0.3f} R = {:0.3f} F1 = {:0.3f}'.format(accuracy, precision, recall, fscore))

    report = seqlab_classification_report(y_true, y_pred)
    print(report)

    return accuracy, precision, recall, fscore

def compute_acc_precision_recall_f1s(y_true, y_pred, target_names, avg='binary'):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=avg)
    recall = recall_score(y_true, y_pred, average=avg)
    fscore = f1_score(y_true, y_pred, average=avg)
    print('Test Metrics: A = {:0.3f} P = {:0.3f} R = {:0.3f} F1 = {:0.3f}'.format(accuracy, precision, recall, fscore))

    report = classification_report(y_true, y_pred, target_names=target_names)
    print(report)

    '''
    By definition, C[i, j] is
        the number of observations known to be in group i
        but predicted to be in group j
    C[0, 0] = true negatives, C[0, 1] = false positives
    C[1, 0] = false negatives, C[1, 1] = true positives
    '''
    matrix = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))
    print(matrix)

    return accuracy, precision, recall, fscore

def read_context_file(page, tag_name):
    '''
        <context-left sent="1">He performed at the Habitat Center, New Delhi on a visit to</context-left>
        <context-left sent="2">Zalog is a formerly independent settlement in the eastern part of the capital</context-left>
        <context-left sent="3"></context-left>
    '''

    infile = open(page, "r")
    xml = infile.read()
    content = BeautifulSoup(xml, 'html5lib')

    lines = list()
    for tag in content.find_all(tag_name):
        lines.append(tag.text.strip())

    return lines

def get_logger(level=logging.DEBUG):
    logger = logging.getLogger(__name__)

    if logger.handlers:
       return logger

    logger.setLevel(level)
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(fmt='%(message)s', datefmt='%m/%d/%Y %I:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

def get_tic():
    return time.time()

def compute_elapsed_time(tic):
    toc = time.time()
    m, s = divmod(toc-tic, 60)
    h, m = divmod(m, 60)
    print('Elapsed Time = %d:%02d:%02d' % (h, m, s))

def dump_to_pickle(file, data):
    if not os.path.exists(file):
        open(file, 'x')
    with open(file, 'wb') as fp:
        pickle.dump(data, fp)

def dump_to_json(file, data):
    if not os.path.exists(file):
        open(file, 'x')
    with open(file, 'w', encoding='utf-8') as fp:
        json.dump(data, fp, ensure_ascii=False, indent=4)

def load_from_pickle(file):
    with open(file, 'rb') as fp:
        data = pickle.load(fp)
    return data

def load_from_json(file):
    with open(file, 'r') as fp:
        data = json.load(fp)
    return data
