import os
import sys
import numpy as np
import argparse

from LSTM_Train import train
from LSTM_Test import test

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--choice', help="base or prewin")
    parser.add_argument('-w', '--window', type=int, help="Window size for baseline")
    parser.add_argument('-d', '--dirname', help="path to directory containing pickle files")
    args = parser.parse_args()
    print(args)

    name = os.path.dirname(args.dirname)
    train(args.choice, name, args.window)
    test(args.choice, name, args.window)
