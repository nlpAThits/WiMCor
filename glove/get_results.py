import os
import sys
import numpy as np
import argparse

from LSTM_Train import train
from LSTM_Test import test

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('choice', help="sp, sp5, ...rp, rp5, ...")
    parser.add_argument('dirname', help="path to directory containing pickle files")
    parser.add_argument('-r', '--repeats', default='1', type=int, help="number of times to repeat the experiment")
    parser.add_argument('-vs', '--val_split', default='1.1', type=float, help="validation split")
    parser.add_argument('-w', '--window', type=int, help="Window size for baseline")
    args = parser.parse_args()
    print(args)

    test_scores = []

    name = os.path.dirname(args.dirname)
    for repeat in range(args.repeats):
        print('Repeat: {}'.format(repeat+1))
        train(args.choice, name, args.val_split, args.window)
        test_scores.append(test(args.choice, name, args.window))

    compute_mean_rpf(test_scores)
