#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import print_function
from __future__ import print_function
from keras import preprocessing
from keras.layers import Convolution1D, MaxPooling1D, LSTM
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import RMSprop
import re
import numpy as np
import math
from keras.utils.visualize_util import plot

# Variables
word_count = 32180
input_length = 2  # word_count must be dividable by this value without any rest
epochs = 5
train_test_ratio = 0.7
# !Variables


def cmp_f(x, y):
    return y[1] - x[1]


def test(text):
    X = []
    X.append(preprocessing.text.one_hot(text, n=word_count))
    capital = 0
    if text.istitle():
        capital = 1
    X.append(preprocessing.text.one_hot(str(capital), n=word_count))
    print("text    :" + text)
    print("encoding:" + str(X))
    predict = np.array(X)
    predict = np.reshape(predict, (-1, input_length))
    print("result  :" + str(model.predict(predict)))


def process_data(X):
    voc_len = len(X)
    R = []
    for t in X:
        encoded = preprocessing.text.one_hot(t[0], n=voc_len)
        R.append([encoded, preprocessing.text.one_hot(t[1], n=voc_len)])
    return R


def load_data(limit=-1):
    with open('../data/prepared/AllFlattenNamWords.csv', 'r') as file:
        X = []
        counter = 0
        for line in file.readlines():
            if re.search("^[.(){},?]+", line) == None:
                x = line.split(",")
                x[1] = int(x[1])
                X.append(x)

                if counter == limit - 1:
                    break;
                counter += 1
        return X


def prepare_data():
    X = load_data(word_count)
    X.sort(cmp=cmp_f)

    npX = np.array(X)
    npY = npX[:, 2]
    npX = npX[:, [0, 1]]
    X = npX.tolist()

    X = process_data(X)
    npX = np.array(X)
    el_len = len(npX[0])
    index = 0
    for element in npX:
        if len(element) != el_len or element == []:
            exit("\n *** ERROR -> different element sizes " + str(element) + " on index = " + str(index) + " *** ")
        index += 1
    train_size = math.floor((len(npX)*train_test_ratio) / input_length) * input_length
    X_train = np.reshape(npX[:train_size], (-1, input_length))
    Y_train = np.reshape(npY[:train_size], (-1, 1))
    X_test = np.reshape(npX[train_size:], (-1, input_length))
    Y_test = np.reshape(npY[train_size:], (-1, 1))
    return X_train, Y_train, X_test, Y_test


X_train, Y_train, X_test, Y_test = prepare_data()

print(X_train.shape)
print(Y_train.shape)

print(X_train[5])

model = Sequential()
model.add(Embedding(word_count, 128, input_length=input_length))
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
model.compile(loss='binary_crossentropy', class_mode='binary', optimizer='rmsprop')

model.fit(X_train, Y_train, verbose=1, nb_epoch=epochs, show_accuracy=True)

print("Training finished")

score, acc = model.evaluate(X_test, Y_test, verbose=1, show_accuracy=True, sample_weight=None)
print('Test score:', score)
print('Test accuracy:', acc)

# print("debug")  # at this point call test(text) to check
