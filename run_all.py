import os
from os import listdir
from os.path import isfile, join

import numpy as np
import scipy.io

import ksvd
from ksvd import ApproximateKSVD
import mlp_utils
import c3d_utils


dataset_name = 'weizmann'
root_dir = '../results/' + dataset_name + '_features/'


## Dictionary learning ##

files = []
for file in os.listdir(root_dir):
    if file.endswith(".mat"):
        files.append(os.path.join(root_dir, file))

# print(files)

# first_time = True
Descr = []
for file in files:
    descr = scipy.io.loadmat(file)['feature']
    print(descr)
    Descr.append(descr)

    # if first_time:
    #     Descr = scipy.io.loadmat(file)['feature']
    #     # print(Descr.shape)
    #     first_time = False
    # else:
    #     tmp = scipy.io.loadmat(file)['feature']
    #     print(tmp.shape)
    #     Descr = np.concatenate((Descr, tmp), axis=0)

print(len(Descr))


## Dictionary Learning:

n_components = 12
for X in Descr:
    aksvd = ApproximateKSVD(n_components=n_components)
    dictionary = aksvd.fit(X).components_
    gamma = aksvd.transform(X)

    print(gamma.shape)


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

