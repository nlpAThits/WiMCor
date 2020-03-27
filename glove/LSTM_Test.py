import sys
import codecs
import random
import re
import copy
import numpy as np
from os import path

from keras.layers import Embedding, TimeDistributed, Flatten, Concatenate, Dense, Dropout, LSTM, Input, concatenate
from keras.models import Sequential, Model
from tensorflow.contrib.keras.api.keras.initializers import Constant

# np.random.seed(133)
# random.seed(133)

from stats import load_from_pickle, dump_to_pickle, compute_acc_precision_recall_f1s

def test(choice, dirname, window):
    #  --------------------------------------------------------------------------------------------------------------------
    dimensionality = 50  # No need to adjust, unless you want to experiment with custom embeddings
    seq_length = 5  # Adjust to 5 for PreWin and 5, 10, 50 for baseline results
    print("Dimensionality:", dimensionality)
    regex = re.compile(r"[+-.]?\d+[-.,\d+:]*(th|st|nd|rd)?")

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
        mlmr_dir = '{}/pickle'.format(dirname)
        seq_length = window  # Adjust to 5 for PreWin and 5, 10, 50 for baseline results

        neg = load_from_pickle("{}/wiki_met_{}{}.pkl".format(dirname, style, base))
        pos = load_from_pickle("{}/wiki_lit_{}{}.pkl".format(dirname, style, base))
    elif choice=='wkmultival' or choice=='wkmultibaseval' or choice=='wkmultitest' or choice=='wkmultibasetest':
        base = ''
        if choice=='wkmultibaseval' or choice=='wkmultibasetest':
            base = '_base'
        style = 'val'
        if choice=='wkmultitest' or choice=='wkmultibasetest':
            style = 'test'
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

    X_L, D_L, X_R, D_R, Y = [], [], [], [], []
    for a in copy.deepcopy(neg + pos):
        X_L.append(a[0][-seq_length:])
        D_L.append(a[1][-seq_length:])
        X_R.append(a[2][:seq_length])
        D_R.append(a[3][:seq_length])
        Y.append(a[4])

    print('No of test examples: ', len(X_L))
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
    foo = Embedding(len(vocabulary), dimensionality, input_length=(seq_length,), embeddings_initializer=Constant(weights))(first_input)
    b = LSTM(units=15)(foo)
    first_output = Dropout(0.2)(b)
    model_left = Model(inputs=first_input, outputs=first_output)

    second_input = Input(shape=(seq_length, len(dep_labels)))
    b = TimeDistributed(Dense(units=15))(second_input)
    c = Dropout(0.2)(b)
    second_output = Flatten()(c)
    dep_left = Model(inputs=second_input, outputs=second_output)

    third_input = Input(shape=(seq_length, ))
    foo = Embedding(len(vocabulary), dimensionality, input_length=(seq_length,), embeddings_initializer=Constant(weights))(third_input)
    b = LSTM(units=15, go_backwards=True)(foo)
    third_output = Dropout(0.2)(b)
    model_right = Model(inputs=third_input, outputs=third_output)

    fourth_input = Input(shape=(seq_length, len(dep_labels)))
    b = TimeDistributed(Dense(units=15))(fourth_input)
    c = Dropout(0.2)(b)
    fourth_output = Flatten()(c)
    dep_right = Model(inputs=fourth_input, outputs=fourth_output)

    a = concatenate([first_output, second_output, third_output, fourth_output])
    b = Dense(10)(a)
    c = Dense(num_classes, activation='softmax')(b)
    merged_model = Model(inputs=[first_input, second_input, third_input, fourth_input], outputs=c)
    merged_model.load_weights("./weights/lstm.hdf5")
    merged_model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])
    print(u"Done...")
    #  --------------------------------------------------------------------------------------------------------------------
    score = merged_model.evaluate([X_L, D_L, X_R, D_R], Y, batch_size=16, verbose=0)
    print('Test accuracy:{:6.3f}'.format(score[1]))
    '''
    name = "conll_base5"
    if True:
        out = codecs.open("./relocar/" + name + ".txt", mode="w", encoding="utf-8")
        for p, y in zip(merged_model.predict_classes([X_L, D_L, X_R, D_R]), Y):
            out.write(str(p[0]) + '\n')
    '''
    #  --------------------------------------------------------------------------------------------------------------------

    predictions = merged_model.predict_on_batch([X_L, D_L, X_R, D_R])
    y_pred = predictions.argmax(axis=1)
    one_hot = np.zeros((y_pred.size, num_classes))
    one_hot[np.arange(y_pred.size), y_pred] = 1
    y_pred = one_hot
    (acc, prec, rec, f1) = compute_acc_precision_recall_f1s(Y, y_pred, avg='micro')
    # assert (str(round(acc, 3)) == str(round(score[1], 3)))

    return acc, prec, rec, f1
