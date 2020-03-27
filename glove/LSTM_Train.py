import sys
import codecs
import random
import re
import numpy as np
from os import path

from keras.callbacks import ModelCheckpoint
from keras.layers import Embedding, TimeDistributed, Flatten, Concatenate, Dense, Dropout, LSTM, Input, concatenate
from keras.models import Sequential, Model
from tensorflow.contrib.keras.api.keras.initializers import Constant

# np.random.seed(133)
# random.seed(133)

from stats import load_from_pickle, dump_to_pickle

def train(choice, dirname, val_split, window=None):
    #  --------------------------------------------------------------------------------------------------------------------
    dimensionality = 50  # No need to adjust, unless you want to experiment with custom embeddings
    print("Dimensionality:", dimensionality)
    regex = re.compile(r"[+-.]?\d+[-.,\d+:]*(th|st|nd|rd)?")
    # Remember to choose the CORRECT file names below otherwise you will see bad things happen :-)
    if choice=='spval' or choice=='spbaseval' or choice=='sptest' or choice=='spbasetest':
        base = ''
        if choice=='spbaseval' or choice=='spbasetest':
            base = '_base'
        style = 'train'
        seq_length = window  # Adjust to 5 for PreWin and 5, 10, 50 for baseline results

        neg = load_from_pickle("{}/semeval_metonymic_{}{}.pkl".format(dirname, style, base))
        ne.extend(load_from_pickle("{}/semeval_mixed_{}{}.pkl".format(dirname, style, base)))
        pos = load_from_pickle("{}/semeval_literal_{}{}.pkl".format(dirname, style, base))
    elif choice=='rcval' or choice=='rcbaseval' or choice=='rctest' or choice=='rcbasetest':
        base = ''
        if choice=='rcbaseval' or choice=='rcbasetest':
            base = '_base'
        style = 'train'
        seq_length = window  # Adjust to 5 for PreWin and 5, 10, 50 for baseline results

        neg = load_from_pickle("{}/relocar_metonymic_train{}.pkl".format(dirname, base))
        pos = load_from_pickle("{}/relocar_literal_train{}.pkl".format(dirname, base))
    elif choice=='orgval' or choice=='orgbaseval' or choice=='orgtest' or choice=='orgbasetest':
        base = ''
        if choice=='orgbaseval' or choice=='orgbasetest':
            base = '_base'
        style = 'train'
        seq_length = window  # Adjust to 5 for PreWin and 5, 10, 50 for baseline results

        neg = load_from_pickle("{}/org_metonymic_train{}.pkl".format(dirname, base))
        neg.extend(load_from_pickle("{}/org_mixed_train{}.pkl".format(dirname, base)))
        pos = load_from_pickle("{}/org_literal_train{}.pkl".format(dirname, base))
    elif choice=='wkbinaryval' or choice=='wkbinarybaseval' or choice=='wkbinarytest' or choice=='wkbinarybasetest':
        base = ''
        if choice=='wkbinarybaseval' or choice=='wkbinarybasetest':
            base = '_base'
        style = 'train'
        mlmr_dir = '{}/pickle'.format(dirname)
        seq_length = window  # Adjust to 5 for PreWin and 5, 10, 50 for baseline results

        neg = load_from_pickle("{}/wiki_met_{}{}.pkl".format(dirname, style, base))
        pos = load_from_pickle("{}/wiki_lit_{}{}.pkl".format(dirname, style, base))
    elif choice=='wkmultival' or choice=='wkmultibaseval' or choice=='wkmultitest' or choice=='wkmultibasetest':
        base = ''
        if choice=='wkmultibaseval' or choice=='wkmultibasetest':
            base = '_base'
        style = 'train'
        mlmr_dir = '{}/pickle'.format(dirname)
        seq_length = window  # Adjust to 5 for PreWin and 5, 10, 50 for baseline results

        neg = load_from_pickle("{}/wiki_INSTITUTE_{}{}.pkl".format(mlmr_dir, style, base))
        if path.exists("{}/wiki_EVENT_{}{}.pkl".format(mlmr_dir, style, base)):
            neg.extend(load_from_pickle("{}/wiki_EVENT_{}{}.pkl".format(mlmr_dir, style, base)))
        if path.exists("{}/wiki_TEAM_{}{}.pkl".format(mlmr_dir, style, base)):
            neg.extend(load_from_pickle("{}/wiki_TEAM_{}{}.pkl".format(mlmr_dir, style, base)))
        if path.exists("{}/wiki_ARTIFACT_{}{}.pkl".format(mlmr_dir, style, base)):
            neg.extend(load_from_pickle("{}/wiki_ARTIFACT_{}{}.pkl".format(mlmr_dir, style, base)))
        pos = load_from_pickle("{}/wiki_LOCATION_{}{}.pkl".format(mlmr_dir, style, base))

    print("Sequence Length: 2 times ", seq_length)

    A = []
    dep_labels = {u"<u>"}
    for coll in [neg, pos]:
        for l in coll:
            A.append(l)
            dep_labels.update(set(l[1][-seq_length:] + l[3][:seq_length]))

    random.shuffle(A)

    X_L, D_L, X_R, D_R, Y = [], [], [], [], []
    for a in A:
        X_L.append(a[0][-seq_length:])
        D_L.append(a[1][-seq_length:])
        X_R.append(a[2][:seq_length])
        D_R.append(a[3][:seq_length])
        Y.append(a[4])

    print('No of training examples: ', len(X_L))
    dump_to_pickle("pickle/dep_labels.pkl", dep_labels)
    dep_labels = load_from_pickle("pickle/dep_labels.pkl")
    #  --------------------------------------------------------------------------------------------------------------------
    vocabulary = {u"<u>", u"0.0"}
    vocab_limit = 100000
    print('Vocabulary Size: ', vocab_limit)
    print("Building sequences...")

    count = 0
    vectors_glove = {u'<u>': np.ones(dimensionality)}
    # Please supply your own embeddings, see README.md for details
    for line in codecs.open("data/glove.6B.50d.txt", encoding="utf-8"):
        tokens = line.split()
        vocabulary.add(tokens[0])
        vectors_glove[tokens[0]] = [float(x) for x in tokens[1:]]
        count += 1
        if count >= vocab_limit:
            break

    vectors_glove[u"0.0"] = np.zeros(dimensionality)
    word_to_index = dict([(w, i) for i, w in enumerate(vocabulary)])
    dep_to_index = dict([(w, i) for i, w in enumerate(dep_labels)])

    for x_l, x_r, d_l, d_r in zip(X_L, X_R, D_L, D_R):
        for i, w in enumerate(x_l):
            if w != u"0.0":
                w = regex.sub(u"1", w)
            if w in word_to_index:
                x_l[i] = word_to_index[w]
            else:
                x_l[i] = word_to_index[u"<u>"]
        for i, w in enumerate(x_r):
            if w != u"0.0":
                w = regex.sub(u"1", w)
            if w in word_to_index:
                x_r[i] = word_to_index[w]
            else:
                x_r[i] = word_to_index[u"<u>"]
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

    X_L = np.asarray(X_L)
    X_R = np.asarray(X_R)
    D_L = np.asarray(D_L)
    D_R = np.asarray(D_R)
    Y = np.asarray(Y)

    # convert labels to one-hot format
    num_classes = Y.max()+1
    one_hot = np.zeros((Y.size, num_classes))
    one_hot[np.arange(Y.size), Y] = 1
    Y = one_hot

    weights = np.zeros((len(vocabulary), dimensionality))
    for w in vocabulary:
        if w in vectors_glove:
            weights[word_to_index[w]] = vectors_glove[w]
    weights = np.array([weights])

    print(u"Done...")
    #  --------------------------------------------------------------------------------------------------------------------
    print(u'Building model...')
    first_input = Input(shape=(seq_length, ))
    a = Embedding(len(vocabulary), dimensionality, input_length=(seq_length,), embeddings_initializer=Constant(weights))(first_input)
    b = LSTM(units=15)(a)
    first_output = Dropout(0.2)(b)
    model_left = Model(inputs=first_input, outputs=first_output)

    second_input = Input(shape=(seq_length, len(dep_labels)))
    a = TimeDistributed(Dense(units=15))(second_input)
    b = Dropout(0.2)(a)
    second_output = Flatten()(b)
    dep_left = Model(inputs=second_input, outputs=second_output)

    third_input = Input(shape=(seq_length, ))
    a = Embedding(len(vocabulary), dimensionality, input_length=(seq_length,), embeddings_initializer=Constant(weights))(third_input)
    b = LSTM(units=15, go_backwards=True)(a)
    third_output = Dropout(0.2)(b)
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
    merged_model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])
    print(u"Done...")
    #  --------------------------------------------------------------------------------------------------------------------
    checkpoint = ModelCheckpoint(filepath="./weights/lstm.hdf5", verbose=0)
    merged_model.fit([X_L, D_L, X_R, D_R], Y, batch_size=16, nb_epoch=5, callbacks=[checkpoint], verbose=0)
    #  --------------------------------------------------------------------------------------------------------------------
