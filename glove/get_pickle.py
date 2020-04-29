import os
import sys
import numpy as np
import argparse

from create_imm import imm
from create_prewin import prewin

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--choice', help="imm or prewin")
    parser.add_argument('-f', '--filepath', help="path to annotated file")
    args = parser.parse_args()
    print(args)

    if args.choice=='imm':
        imm(args.filepath)
    elif args.choice=='prewin':
        prewin(args.filepath)

