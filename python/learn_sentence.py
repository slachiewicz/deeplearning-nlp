#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import print_function
from __future__ import print_function
from keras import preprocessing
from keras.layers import Convolution1D, MaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, Graph
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import RMSprop
import re
import numpy as np
import math
import sys

# Variables
word_count = 32180 #w tym przypadku chyba mniej slow
epochs = 10
train_test_ratio = 0.7
dropout_prob = (0.25, 0.5)
hidden_dims = 150
embedding_weights = None
embedding_dim = 20
filter_sizes=(3, 4)
num_filters = 150

# !Variables


def test(text, input_length, model):
    X = []
    prep = preprocessing.text.one_hot(text, n=word_count, split=" ")
    print("text :" + text)
    while len(prep) < input_length:
        prep.append(0)
    X.append(prep)
    print("encoding:" + str(X))
    predict = np.array(X)
    print(predict)
    print("result  :" + str(model.predict(predict)))


def process_sentence(sen):
    return preprocessing.text.one_hot(sen, n=word_count, split=" ")

def adjust_data(X, size):
    N = []
    for sen in X:
        while len(sen) < size:
            sen.append(0)
        N.append(sen)
    return N


def load_data(limit=-1):
    with open('../data/prepared/AllFlattenNamSentences.csv', 'r') as file:
        X = []
        Y = []
        counter = 0
        longest_sentence = 0

        for line in file.readlines():
            data = line.rstrip(";\n").split(",")
            en_words = process_sentence(data[0])
            values = [int(i) for i in data[1].split(';')]

            if len(values) > longest_sentence:
                longest_sentence = len(values)

            X.append(en_words)
            Y.append(values)

            if counter == limit - 1:
                break
            counter += 1

        X = adjust_data(X, longest_sentence)
        Y = adjust_data(Y, longest_sentence)

        return np.array(X), np.array(Y)

def prepare_data(X, Y):

    train_size = math.floor((X.shape[0] * train_test_ratio))

    print("Train size: {}".format(train_size))

    X_train = X[:train_size]
    Y_train = Y[:train_size]
    X_test = X[train_size:]
    Y_test = Y[train_size:]
    return X_train, Y_train, X_test, Y_test

X, Y = load_data()
X_train, Y_train, X_test, Y_test = prepare_data(X, Y)

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)

graph = Graph()
graph.add_input(name='input', input_shape=(X_train.shape[1], embedding_dim))
for fsz in filter_sizes:
    conv = Convolution1D(nb_filter=num_filters,
                         filter_length=fsz,
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1)
    pool = MaxPooling1D(pool_length=2)
    graph.add_node(conv, name='conv-%s' % fsz, input='input')
    graph.add_node(pool, name='maxpool-%s' % fsz, input='conv-%s' % fsz)
    graph.add_node(Flatten(), name='flatten-%s' % fsz, input='maxpool-%s' % fsz)
graph.add_output(name='output', inputs=['flatten-%s' % fsz for fsz in filter_sizes], merge_mode='concat')

model = Sequential()
model.add(Embedding(word_count, embedding_dim, input_length=X_train.shape[1], weights=embedding_weights))
model.add(Dropout(dropout_prob[0], input_shape=(X_train.shape[1], embedding_dim)))
model.add(graph)
model.add(Dense(hidden_dims))
model.add(Dropout(dropout_prob[1]))
model.add(Activation('relu'))
model.add(Dense(X_train.shape[1]))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', class_mode='binary')

#rms = RMSprop()
#model.compile(loss='binary_crossentropy', optimizer='rmsprop')

model.fit(X_train, Y_train, verbose=1, nb_epoch=epochs)

print("Training finished")

result = model.evaluate(X_test, Y_test, verbose=1, sample_weight=None)
print("Testing result: " + str(result))

print("debug")  # at this point call test(text) to check
test("Ala ma kota", X_train.shape[1], model)
#json = model.to_json()
#open("model_json", "w").write(json)
#model.save_weights("model_weights.h5", overwrite=True)

