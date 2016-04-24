#!/usr/bin/python
# -*- coding: utf-8 -*-

from keras import preprocessing
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence, text
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
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
    with open('../data/prepared/AllFlattenNamWords.txt', 'r') as file:
        X = []
        counter = 0
        for line in file.readlines():
            if re.search("^[.(){},?]+", line) == None:
                x = line.split(" ")
                x[1] = int(x[1])
                X.append(x)

                if counter == limit-1:
                    break;
                counter += 1
        return X

X = load_data(500)
X.sort(cmp=cmp_f)

npX = np.array(X)
npY = npX[:,1]
npX = npX[:,0]
X = npX.tolist()

X = process_data(X)
npX = np.array(X)
print npX.shape
print npY.shape

print np.random.random((5,10))

model = Sequential()
model.add(Dense(1, input_shape=(1,)))
model.add(Activation('relu'))

rms = RMSprop()
model.compile(loss='categorical_crossentropy', optimizer=rms)

#npY = np.transpose(npY)

model.fit(npX, npY)


#token = Tokenizer(nb_words=word)
#token.fit_on_texts(lista)
#sequences = np.array(token.texts_to_sequences(lista))
#print sequences.shape

#x = "Mark Zukerberg and his wife to donate $120M to needy Bay Area schools"
#
#tok = Tokenizer(nb_words=13, lower=False)
#tok.fit_on_texts(x)
#
#data = tok.texts_to_sequences(x)
#print data
#
#for i in tok.texts_to_sequences_generator(x):
#    couples, labels = sequence.skipgrams(i, 12, window_size=2, negative_samples=0.)
#    print couples, labels
#
#model = Sequential()
#model.add(Embedding(input_dim=1, output_dim=2))
#
#model.compile(optimizer='adam', loss='binary_crossentropy')


