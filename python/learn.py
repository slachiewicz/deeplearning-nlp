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


def cmp_f(x, y):
    return y[1] - x[1]


def process_data(X):
    voc_len = len(set(X))
    R = []
    for t in X:
        R.append(preprocessing.text.one_hot(t, n=voc_len))
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
    npY = npX[:, 1]
    npX = npX[:, 0]
    X = npX.tolist()

    X = process_data(X)
    npX = np.array(X)
    el_len = len(npX[0])
    index = 0
    for element in npX:
        if len(element) != el_len or element == []:
            exit("\n *** ERROR -> different element sizes " + str(element) + " on index = " + str(index) + " *** ")
        index += 1

    npX = np.reshape(npX, (-1, input_length))
    npY = np.reshape(npY, (-1, input_length))
    return npX, npY


# Variables
word_count = 32180
input_length = 10  # word_count must be dividable by this value without any rest
epochs = 10
# !Variables

npX, npY = prepare_data()

print(npX.shape)
print(npY.shape)

print(np.random.random((5, 10)))

model = Sequential()
model.add(Embedding(word_count, 128, input_length=input_length))
model.add(Dropout(0.25))

# we add a Convolution1D, which will learn nb_filter
# word group filters of size filter_length:
model.add(Convolution1D(nb_filter=250,
                        filter_length=3,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
# we use standard max pooling (halving the output of the previous layer):
model.add(MaxPooling1D(pool_length=2))
model.add(Flatten())

# We add a vanilla hidden layer:
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
model.compile(loss='categorical_crossentropy', optimizer=rms)

model.fit(npX, npY, nb_epoch=epochs)

# token = Tokenizer(nb_words=word)
# token.fit_on_texts(lista)
# sequences = np.array(token.texts_to_sequences(lista))
# print sequences.shape

# x = "Mark Zukerberg and his wife to donate $120M to needy Bay Area schools"
#
# tok = Tokenizer(nb_words=13, lower=False)
# tok.fit_on_texts(x)
#
# data = tok.texts_to_sequences(x)
# print data
#
# for i in tok.texts_to_sequences_generator(x):
#    couples, labels = sequence.skipgrams(i, 12, window_size=2, negative_samples=0.)
#    print couples, labels
#
# model = Sequential()
# model.add(Embedding(input_dim=1, output_dim=2))
#
# model.compile(optimizer='adam', loss='binary_crossentropy')
