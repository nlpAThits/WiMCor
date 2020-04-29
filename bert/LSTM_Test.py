import codecs
import re
import copy
import numpy as np
from os import path

import torch
from keras.layers import Embedding
from keras.layers import TimeDistributed
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Input
from keras.layers import concatenate
from keras.models import Model
from sklearn.metrics import precision_recall_fscore_support
from tensorflow.contrib.keras.api.keras.initializers import Constant

from utils import load_from_pickle, dump_to_pickle
from LSTM_Train import load_from_hdf5

BERT_DIMENSIONS = 768

def test(choice, dirname, window):
    #  --------------------------------------------------------------------------------------------------------------------
    dimensionality = BERT_DIMENSIONS  # No need to adjust, unless you want to experiment with custom embeddings
    seq_length = 5  # Adjust to 5 for PreWin and 5, 10, 50 for baseline results
    print("Dimensionality:", dimensionality)

    if choice=='imm':
        base = '_base'
    elif choice=='prewin':
        base = ''
    style = 'test'
    mlmr_dir = dirname
    seq_length = window  # Adjust to 5 for PreWin and 5, 10, 50 for baseline results

    neg = load_from_hdf5("{}/wiki_LOCATION_{}{}.hdf5".format(mlmr_dir, style, base))
    pos = load_from_hdf5("{}/wiki_INSTITUTE_{}{}.hdf5".format(mlmr_dir, style, base))
    if path.exists("{}/wiki_EVENT_{}{}.hdf5".format(mlmr_dir, style, base)):
        pos.extend(load_from_hdf5("{}/wiki_EVENT_{}{}.hdf5".format(mlmr_dir, style, base)))
    if path.exists("{}/wiki_TEAM_{}{}.hdf5".format(mlmr_dir, style, base)):
        pos.extend(load_from_hdf5("{}/wiki_TEAM_{}{}.hdf5".format(mlmr_dir, style, base)))
    if path.exists("{}/wiki_ARTIFACT_{}{}.hdf5".format(mlmr_dir, style, base)):
        pos.extend(load_from_hdf5("{}/wiki_ARTIFACT_{}{}.hdf5".format(mlmr_dir, style, base)))

    print("Sequence Length: 2 times ", seq_length)

    D_L, E_L, D_R, E_R, Y = [], [], [], [], []
    for a in copy.deepcopy(neg + pos):
        D_L.append(a[0][-seq_length:])
        E_L.append(a[1][-seq_length:])
        D_R.append(a[2][:seq_length])
        E_R.append(a[3][:seq_length])
        Y.append(a[4])

    print('No of test examples: ', len(D_L))
    dep_labels = load_from_pickle("dep_labels.pkl")
    #  --------------------------------------------------------------------------------------------------------------------
    print("Building sequences...")

    dep_to_index = dict([(w, i) for i, w in enumerate(dep_labels)])

    for d_l, d_r in zip(D_L, D_R):
        for i, w in enumerate(d_l):
            arr = np.zeros(len(dep_labels))
            if w in dep_to_index:
                arr[dep_to_index[w]] = 1
            else:
                arr[dep_to_index[u"<u>"]] = 1
            d_l[i] = arr
        for i, w in enumerate(d_r):
            arr = np.zeros(len(dep_labels))
            if w in dep_to_index:
                arr[dep_to_index[w]] = 1
            else:
                arr[dep_to_index[u"<u>"]] = 1
            d_r[i] = arr

    D_L = np.asarray(D_L)
    D_R = np.asarray(D_R)
    E_L = torch.stack(E_L).detach()
    E_R = torch.stack(E_R).detach()
    Y = np.asarray(Y)

    # convert labels to one-hot format
    num_classes = Y.max()+1
    one_hot = np.zeros((Y.size, num_classes))
    one_hot[np.arange(Y.size), Y] = 1
    Y = one_hot

    print(u"Done...")
    #  --------------------------------------------------------------------------------------------------------------------
    print(u'Building model...')
    first_input = Input(shape=(seq_length, dimensionality))
    a = LSTM(units=15)(first_input)
    first_output = Dropout(0.2)(a)
    model_left = Model(inputs=first_input, outputs=first_output)

    second_input = Input(shape=(seq_length, len(dep_labels)))
    a = TimeDistributed(Dense(units=15))(second_input)
    b = Dropout(0.2)(a)
    second_output = Flatten()(b)
    dep_left = Model(inputs=second_input, outputs=second_output)

    third_input = Input(shape=(seq_length, dimensionality))
    a = LSTM(units=15, go_backwards=True)(third_input)
    third_output = Dropout(0.2)(a)
    model_right = Model(inputs=third_input, outputs=third_output)

    fourth_input = Input(shape=(seq_length, len(dep_labels)))
    a = TimeDistributed(Dense(units=15))(fourth_input)
    b = Dropout(0.2)(a)
    fourth_output = Flatten()(b)
    dep_right = Model(inputs=fourth_input, outputs=fourth_output)

    a = concatenate([first_output, second_output, third_output, fourth_output])
    b = Dense(10)(a)
    c = Dense(num_classes, activation='softmax')(b)
    merged_model = Model(inputs=[first_input, second_input, third_input, fourth_input], outputs=c)
    merged_model.load_weights("lstm.hdf5")
    merged_model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])
    print(u"Done...")
    #  --------------------------------------------------------------------------------------------------------------------
    predictions = merged_model.predict_on_batch([E_L, D_L, E_R, D_R])
    y_pred = predictions.argmax(axis=1)
    one_hot = np.zeros((y_pred.size, num_classes))
    one_hot[np.arange(y_pred.size), y_pred] = 1
    y_pred = one_hot

    print('Macro-averaged metrics: ', precision_recall_fscore_support(Y, y_pred, average='macro'))
    print('Micro-averaged metrics: ', precision_recall_fscore_support(Y, y_pred, average='micro'))
