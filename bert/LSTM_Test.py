import sys
import codecs
import random
import re
import copy
import numpy as np
from os import path

import torch
from keras.layers import Embedding, TimeDistributed, Flatten, Concatenate, Dense, Dropout, LSTM, Input, concatenate
from keras.models import Sequential, Model
from tensorflow.contrib.keras.api.keras.initializers import Constant

# np.random.seed(133)
# random.seed(133)

from stats import load_from_pickle, dump_to_pickle, compute_acc_precision_recall_f1s
from LSTM_Train import load_from_hdf5

BERT_DIMENSIONS = 768

def test(choice, dirname, window):
    #  --------------------------------------------------------------------------------------------------------------------
    dimensionality = BERT_DIMENSIONS  # No need to adjust, unless you want to experiment with custom embeddings
    seq_length = 5  # Adjust to 5 for PreWin and 5, 10, 50 for baseline results
    print("Dimensionality:", dimensionality)

    if choice=='spval' or choice=='spbaseval' or choice=='sptest' or choice=='spbasetest':
        base = ''
        if choice=='spbaseval' or choice=='spbasetest':
            base = '_base'
        style = 'val'
        if choice=='sptest' or choice=='spbasetest':
            style = 'test'
        seq_length = window  # Adjust to 5 for PreWin and 5, 10, 50 for baseline results

        neg = load_from_pickle("{}/semeval_metonymic_{}{}.pkl".format(dirname, style, base))
        ne.extend(load_from_pickle("{}/semeval_mixed_{}{}.pkl".format(dirname, style, base)))
        pos = load_from_pickle("{}/semeval_literal_{}{}.pkl".format(dirname, style, base))
    elif choice=='rcval' or choice=='rcbaseval' or choice=='rctest' or choice=='rcbasetest':
        base = ''
        if choice=='rcbaseval' or choice=='rcbasetest':
            base = '_base'
        style = 'val'
        if choice=='rctest' or choice=='rcbasetest':
            style = 'test'
        seq_length = window  # Adjust to 5 for PreWin and 5, 10, 50 for baseline results

        neg = load_from_pickle("{}/relocar_metonymic_train{}.pkl".format(dirname, base))
        pos = load_from_pickle("{}/relocar_literal_train{}.pkl".format(dirname, base))
    elif choice=='orgval' or choice=='orgbaseval' or choice=='orgtest' or choice=='orgbasetest':
        base = ''
        if choice=='orgbaseval' or choice=='orgbasetest':
            base = '_base'
        style = 'val'
        if choice=='orgtest' or choice=='orgbasetest':
            style = 'test'
        seq_length = window  # Adjust to 5 for PreWin and 5, 10, 50 for baseline results

        neg = load_from_pickle("{}/org_metonymic_train{}.pkl".format(dirname, base))
        neg.extend(load_from_pickle("{}/org_mixed_train{}.pkl".format(dirname, base)))
        pos = load_from_pickle("{}/org_literal_train{}.pkl".format(dirname, base))
    elif choice=='wkbinaryval' or choice=='wkbinarybaseval' or choice=='wkbinarytest' or choice=='wkbinarybasetest':
        base = ''
        if choice=='wkbinarybaseval' or choice=='wkbinarybasetest':
            base = '_base'
        style = 'val'
        if choice=='wkbinarytest' or choice=='wkbinarybasetest':
            style = 'test'
        mlmr_dir = '{}/bert_pickle'.format(dirname)
        seq_length = window  # Adjust to 5 for PreWin and 5, 10, 50 for baseline results

        neg = load_from_hdf5("{}/wiki_met_{}{}.hdf5".format(dirname, style, base))
        pos = load_from_hdf5("{}/wiki_lit_{}{}.hdf5".format(dirname, style, base))
    elif choice=='wkmultiprewval' or choice=='wkmultibaseval' or choice=='wkmultiprewtest' or choice=='wkmultibasetest':
        base = ''
        if choice=='wkmultibaseval' or choice=='wkmultibasetest':
            base = '_base'
        style = 'val'
        if choice=='wkmultiprewtest' or choice=='wkmultibasetest':
            style = 'test'
        mlmr_dir = '{}/bert_pickle'.format(dirname)
        seq_length = window  # Adjust to 5 for PreWin and 5, 10, 50 for baseline results

        neg = load_from_hdf5("{}/wiki_INSTITUTE_{}{}.hdf5".format(mlmr_dir, style, base))
        if path.exists("{}/wiki_EVENT_{}{}.hdf5".format(mlmr_dir, style, base)):
            neg.extend(load_from_hdf5("{}/wiki_EVENT_{}{}.hdf5".format(mlmr_dir, style, base)))
        if path.exists("{}/wiki_TEAM_{}{}.hdf5".format(mlmr_dir, style, base)):
            neg.extend(load_from_hdf5("{}/wiki_TEAM_{}{}.hdf5".format(mlmr_dir, style, base)))
        if path.exists("{}/wiki_ARTIFACT_{}{}.hdf5".format(mlmr_dir, style, base)):
            neg.extend(load_from_hdf5("{}/wiki_ARTIFACT_{}{}.hdf5".format(mlmr_dir, style, base)))
        pos = load_from_hdf5("{}/wiki_LOCATION_{}{}.hdf5".format(mlmr_dir, style, base))

    print("Sequence Length: 2 times ", seq_length)

    D_L, E_L, D_R, E_R, Y = [], [], [], [], []
    for a in copy.deepcopy(neg + pos):
        D_L.append(a[0][-seq_length:])
        E_L.append(a[1][-seq_length:])
        D_R.append(a[2][:seq_length])
        E_R.append(a[3][:seq_length])
        Y.append(a[4])

    print('No of test examples: ', len(D_L))
    dep_labels = load_from_pickle("bert_pickle/dep_labels.pkl")
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
    merged_model.load_weights("./weights/lstm.hdf5")
    merged_model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])
    print(u"Done...")
    #  --------------------------------------------------------------------------------------------------------------------
    score = merged_model.evaluate([E_L, D_L, E_R, D_R], Y, batch_size=16, verbose=1)
    print('Test accuracy:{:6.3f}'.format(score[1]))
    '''
    name = "conll_base5"
    if True:
        out = codecs.open("./relocar/" + name + ".txt", mode="w", encoding="utf-8")
        for p, y in zip(merged_model.predict_classes([X_L, D_L, X_R, D_R]), Y):
            out.write(str(p[0]) + '\n')
    '''
    #  --------------------------------------------------------------------------------------------------------------------

    predictions = merged_model.predict_on_batch([E_L, D_L, E_R, D_R])
    y_pred = predictions.argmax(axis=1)
    one_hot = np.zeros((y_pred.size, num_classes))
    one_hot[np.arange(y_pred.size), y_pred] = 1
    y_pred = one_hot
    (acc, prec, rec, f1) = compute_acc_precision_recall_f1s(Y, y_pred, avg='micro')
    # assert (str(round(acc, 3)) == str(round(score[1], 3)))

    return acc, prec, rec, f1
