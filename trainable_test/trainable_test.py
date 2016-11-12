#!/usr/bin/env python

from keras.models import Sequential
from keras.layers import Dense
import numpy as np

def set_trainable(model, trainable):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable

samples = 1000
X1 = np.random.uniform(size=[samples, 100])
X2 = np.random.uniform(size=[samples, 10])

modelA = Sequential([
    Dense(10, input_dim=100, activation='sigmoid')
])

modelB = Sequential([
    Dense(100, input_dim=10, activation='sigmoid')
])

modelB.compile(optimizer='adam', loss='binary_crossentropy')

set_trainable(modelB, False)
connected = Sequential([modelA, modelB])
connected.compile(optimizer='adam', loss='binary_crossentropy')


w0 = np.copy(modelB.layers[0].get_weights()[0])

connected.fit(X1, X1)
w1 = np.copy(modelB.layers[0].get_weights()[0])
print('Freezed in "connected":', np.array_equal(w0, w1))

modelB.fit(X2, X1)
w2 = np.copy(modelB.layers[0].get_weights()[0])
print('Freezed in "modelB":', np.array_equal(w1, w2))