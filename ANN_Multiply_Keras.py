# The code is loosely based on the example in
# https://towardsdatascience.com/building-our-first-neural-network-in-keras-bdc8abbc17f5
#
# Copyright © 2019 Ehsan Khatami

import numpy as np
import matplotlib.pyplot as plt

import tensorflow.python.keras

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

# Generate data and parameters.
# Data sizes
N = 20000

# Network parameters
N_in  = 2
N_hid = 30
N_out = 2

# Generating the data (two number to be multiplied) and their labels (their product)
X = np.random.random(size=2*N).reshape(N,2)

Y = np.zeros([N,2])
Y[:,0] = X[:,0]*X[:,1]

# This is becuase the output has two neurons and we use softmax
Y[:,1] = 1 - Y[:,0]

# Creating training and testing data sets
X_train = X[:N*8//10,:]
X_test = X[N*8//10:,:]
Y_train = Y[:N*8//10,:]
Y_test = Y[N*8//10:,:]

# Build the neural network
model = Sequential()
model.add(Dense(N_hid, input_dim=N_in, activation='sigmoid'))
model.add(Dense(10, activation='sigmoid'))
model.add(Dense(N_out, activation='softmax'))
# Specify the loss function, the optimizer and how we want to measure the success
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])

# Train the network using the fit option
history = model.fit(X_train, Y_train, validation_data = (X_test,Y_test), epochs=100, batch_size=100)

print(list(history.history.keys()))

# Plot the results for model accuracy.
fig, ax = plt.subplots()
ax.set_title('Model Accuracy')
ax.plot(history.history['accuracy'])
ax.plot(history.history['val_accuracy'])
ax.set_ylabel('Accuracy')
ax.set_xlabel('Epoch')
ax.legend(['Train', 'Test'], loc='lower right')

# Plot the results for model loss.
fig2, ax2 = plt.subplots()
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('Model loss')
ax2.set_ylabel('Loss')
ax2.set_xlabel('Epoch')
ax2.legend(['Train', 'Test'], loc='upper right')

# Plot comparison between expected and training data
N = 1000
X  = np.random.random(size=2*N).reshape(N,2)
Y = np.zeros([N,2])
Y[:,0] = X[:,0]*X[:,1]
Y[:,1] = 1 - Y[:,0]

y_pred = model.predict(X)

fig3, ax3 = plt.subplots()
ax3.plot([0,1], [0,1],'r-', linewidth=2, label='Expected')
ax3.plot(Y[:, 0], y_pred[:,0],'o',markersize=2, label='Actual')
ax3.set_xlabel('Label (actual product of 2 numbers)')
ax3.set_ylabel('Result')
ax3.legend()

plt.show()