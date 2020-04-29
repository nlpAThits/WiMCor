import codecs
import re
import copy
import numpy as np
from os import path

from keras.layers import Embedding
from keras.layers import TimeDistributed
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Input
from keras.layers import concatenate
from keras.models import Sequential, Model
from sklearn.metrics import precision_recall_fscore_support
from tensorflow.contrib.keras.api.keras.initializers import Constant

from utils import load_from_pickle, dump_to_pickle

def test(choice, dirname, window):
    #  --------------------------------------------------------------------------------------------------------------------
    dimensionality = 50  # No need to adjust, unless you want to experiment with custom embeddings
    print("Dimensionality:", dimensionality)
    regex = re.compile(r"[+-.]?\d+[-.,\d+:]*(th|st|nd|rd)?")

    if choice=='imm':
        base = '_imm'
    elif choice=='prewin':
        base = ''
    style = 'test'
    mlmr_dir = dirname
    seq_length = window  # Adjust to 5 for PreWin and 5, 10, 50 for baseline results

    neg = load_from_pickle("{}/wiki_LOCATION_{}{}.pkl".format(mlmr_dir, style, base))
    pos = load_from_pickle("{}/wiki_INSTITUTE_{}{}.pkl".format(mlmr_dir, style, base))
    if path.exists("{}/wiki_EVENT_{}{}.pkl".format(mlmr_dir, style, base)):
        pos.extend(load_from_pickle("{}/wiki_EVENT_{}{}.pkl".format(mlmr_dir, style, base)))
    if path.exists("{}/wiki_TEAM_{}{}.pkl".format(mlmr_dir, style, base)):
        pos.extend(load_from_pickle("{}/wiki_TEAM_{}{}.pkl".format(mlmr_dir, style, base)))
    if path.exists("{}/wiki_ARTIFACT_{}{}.pkl".format(mlmr_dir, style, base)):
        pos.extend(load_from_pickle("{}/wiki_ARTIFACT_{}{}.pkl".format(mlmr_dir, style, base)))

    print("Sequence Length: 2 times ", seq_length)

    X_L, D_L, X_R, D_R, Y = [], [], [], [], []
    for a in copy.deepcopy(neg + pos):
        X_L.append(a[0][-seq_length:])
        D_L.append(a[1][-seq_length:])
        X_R.append(a[2][:seq_length])
        D_R.append(a[3][:seq_length])
        Y.append(a[4])

    print('No of test examples: ', len(X_L))
    dep_labels = load_from_pickle("dep_labels.pkl")
    #  --------------------------------------------------------------------------------------------------------------------
    vocabulary = {u"<u>", u"0.0"}
    vocab_limit = 100000
    print('Vocabulary Size: ', vocab_limit)
    print("Building sequences...")

    count = 0
    vectors_glove = {u'<u>': np.ones(dimensionality)}
    # Please supply your own embeddings, see README.md for details
    for line in codecs.open("glove.6B.50d.txt", encoding="utf-8"):
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
    foo = Embedding(len(vocabulary), 
                    dimensionality, 
                    input_length=(seq_length,), 
                    embeddings_initializer=Constant(weights))(first_input)
    b = LSTM(units=15)(foo)
    first_output = Dropout(0.2)(b)
    model_left = Model(inputs=first_input, outputs=first_output)

    second_input = Input(shape=(seq_length, len(dep_labels)))
    b = TimeDistributed(Dense(units=15))(second_input)
    c = Dropout(0.2)(b)
    second_output = Flatten()(c)
    dep_left = Model(inputs=second_input, outputs=second_output)

    third_input = Input(shape=(seq_length, ))
    foo = Embedding(len(vocabulary), 
                    dimensionality, 
                    input_length=(seq_length,), 
                    embeddings_initializer=Constant(weights))(third_input)
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
    merged_model.load_weights("lstm.hdf5")
    merged_model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])
    print(u"Done...")
    #  --------------------------------------------------------------------------------------------------------------------
    predictions = merged_model.predict_on_batch([X_L, D_L, X_R, D_R])
    y_pred = predictions.argmax(axis=1)
    one_hot = np.zeros((y_pred.size, num_classes))
    one_hot[np.arange(y_pred.size), y_pred] = 1
    y_pred = one_hot

    print('Macro-averaged metrics: ', precision_recall_fscore_support(Y, y_pred, average='macro'))
    print('Micro-averaged metrics: ', precision_recall_fscore_support(Y, y_pred, average='micro'))

