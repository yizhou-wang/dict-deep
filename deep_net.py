import os
from os import listdir
from os.path import isfile, join

import numpy as np
import scipy.io
from sklearn.utils import shuffle

import keras
import mlp_utils
import c3d_utils


dataset_name = 'weizmann'
weizmann_label_dic = [ 'bend', 'jack', 'jump', 'pjump', 'run', 'side', 'skip', 'walk', 'wave1', 'wave2' ]
weizmann_label_num = len(weizmann_label_dic)


## Load Dic Results ##
print('Loading Dic Results ...')
root_dir = '../results/' + dataset_name + '_dic/'
files = []
labels_name = []
for file in sorted(os.listdir(root_dir)):
    if file.endswith(".mat"):
        files.append(os.path.join(root_dir, file))
        labels_name.append(file.split('.')[0])


for file, label in zip(files, labels_name):
    
    X1 = scipy.io.loadmat(file)['X']
    try:
        X = np.concatenate((X, X1), axis=0)
    except:
        X = X1
    
    try:
        label_num = weizmann_label_dic.index(label)
    except:
        label_num = weizmann_label_dic.index(label[0:-1])

    fea_num = X1.shape[0]
    Y1 = np.full((fea_num, 1), label_num, dtype=int)
    try:
        Y = np.concatenate((Y, Y1), axis=0)
    except:
        Y = Y1

print('X.shape:', X.shape)
print('Y.shape:', Y.shape)

## Put sparse features into simple neural networks ##

X, Y = shuffle(X, Y, random_state=0)
# print('X.shape:', X.shape)
# print('Y.shape:', Y.shape)

model_mlp = mlp_utils.get_model(summary=True)
model_mlp.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# # Generate dummy data
# data = np.random.random((1000, 100))
# labels = np.random.randint(10, size=(1000, 1))

# Convert labels to categorical one-hot encoding
one_hot_labels = keras.utils.to_categorical(Y, num_classes=10)

# Train the model, iterating on the data in batches of 32 samples
model_mlp.fit(X, one_hot_labels, epochs=100, batch_size=32)



## Put original video into C3D networks ##
# model_c3d = c3d_utils.get_model(summary=True)



## Evaluation ##