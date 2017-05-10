import os
from os import listdir
from os.path import isfile, join

import numpy as np
from scipy.io import loadmat

import ksvd
from ksvd import ApproximateKSVD
import mlp_utils
import c3d_utils


## Initialization ##


## Read videos ##


## Feature extraction ##

files = []
for file in os.listdir('./Feature_seq'):
    if file.endswith(".mat"):
        files.append(os.path.join('./Feature_seq', file))

print(files)

first_time = True
for file in files:
    if first_time:
        Descr = np.loadtxt(file)
        print(Descr.shape)
        first_time = False
    else:
        tmp = np.loadtxt(file)
        print(tmp.shape)
        Descr = np.concatenate((Descr, tmp), axis=0)

print(Descr.shape)
# print(Descr)


# ## Dictionary learning ##

# # SAMPLE CODE FOR Dictionary Learning:
# X = Training data 
# n_components = param
# aksvd = ApproximateKSVD(n_components=128)
# dictionary = aksvd.fit(X).components_
# gamma = aksvd.transform(X)


# ## Put sparse features into simple neural networks ##

# model_mlp = mlp_utils.get_model(summary=True)
# model_mlp.compile(optimizer='rmsprop',
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])

# # Generate dummy data
# data = np.random.random((1000, 100))
# labels = np.random.randint(10, size=(1000, 1))

# # Convert labels to categorical one-hot encoding
# one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)

# # Train the model, iterating on the data in batches of 32 samples
# model.fit(data, one_hot_labels, epochs=10, batch_size=32)



# ## Put original video into C3D networks ##
# model_c3d = c3d_utils.get_model(summary=True)



# ## Evaluation ##

