#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import print_function
from __future__ import print_function
from keras import preprocessing
from keras.layers import Convolution1D, MaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import RMSprop
import re
import numpy as np
import math
import sys

# Variables
word_count = 32180
epochs = 10
train_test_ratio = 0.7
# !Variables


def test(text):
    X = []
    X.append(preprocessing.text.one_hot(text, n=word_count, split=" "))
    print("text    :" + text)
    print("encoding:" + str(X))
    predict = np.array(X)
    predict = np.reshape(predict, (-1, input_length))
    print(predict)
    #print("result  :" + str(model.predict(predict)))


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
        longes_sentence = 0

        for line in file.readlines():
            data = line.rstrip(";\n").split(",")
            en_words = process_sentence(data[0])
            values = [int(i) for i in data[1].split(';')]

            if len(en_words) > longes_sentence:
                longes_sentence = len(en_words)

            X.append(en_words)
            Y.append(values)

            if counter == limit - 1:
                break
            counter += 1

        X = adjust_data(X, longes_sentence)
        Y = adjust_data(Y, longes_sentence)

        return np.array(X), np.array(Y)

def prepare_data():
    X, Y = load_data(5)
    print(X.shape)
    print(Y.shape)

    train_size = math.floor((X.shape[0] * train_test_ratio))

    print("Train size: {}".format(train_size))

    X_train = X[:train_size]
    Y_train = Y[:train_size]
    X_test = X[train_size:]
    Y_test = Y[train_size:]
    return X_train, Y_train, X_test, Y_test


X_train, Y_train, X_test, Y_test = prepare_data()

print(X_train.shape)
print(Y_train.shape)
print(X_train)
print(Y_train)

sys.exit("")

model = Sequential()
model.add(Embedding(word_count, 128, input_length=X_train.shape[1]))
model.add(Dropout(0.25))

# we add a Convolution1D, which will learn nb_filter
# word group filters of size filter_length:
model.add(Convolution1D(nb_filter=250,
                        filter_length=1,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
# we use standard max pooling (halving the output of the previous layer):
model.add(MaxPooling1D(pool_length=1))
model.add(Flatten())

# # We add a vanilla hidden layer:
model.add(Dense(250))
model.add(Dropout(0.25))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1))
model.add(Activation('sigmoid'))

# model.compile(loss='binary_crossentropy',
#               optimizer='rmsprop')
# model.add(Embedding(1,2))
# model.add(Dense(1, input_shape=(1,)))
# model.add(Activation('relu'))

rms = RMSprop()
# model.compile(loss='categorical_crossentropy', optimizer=rms)
model.compile(loss='binary_crossentropy', optimizer='rmsprop')

model.fit(X_train, Y_train, verbose=1, nb_epoch=epochs)

print("Training finished")

result = model.evaluate(X_test, Y_test, verbose=1, sample_weight=None)
print("Testing result: " + str(result))

print("debug")  # at this point call test(text) to check

