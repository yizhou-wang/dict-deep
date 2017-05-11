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
transform_n_nonzero_coefs = 20

## Load Dic Results ##
print('Loading Dic Results ...')
root_dir = '../results/' + dataset_name + '_dic_' + str(transform_n_nonzero_coefs) + '/'
files = []
labels_name = []
for file in sorted(os.listdir(root_dir)):
    if file.endswith(".mat"):
        files.append(os.path.join(root_dir, file))
        labels_name.append(file.split('.')[0])


for file, label in zip(files, labels_name): # NUM of loops: 10
    
    print(file)
    print(label)

    X1 = scipy.io.loadmat(file)['X_train']
    X2 = scipy.io.loadmat(file)['X_test']
    try:
        X_train = np.concatenate((X_train, X1), axis=0)
    except:
        X_train = X1
    try:
        X_test = np.concatenate((X_test, X2), axis=0)
    except:
        X_test = X2
   
    try:
        label_num = weizmann_label_dic.index(label)
    except:
        label_num = weizmann_label_dic.index(label[0:-1])

    tr_num = X1.shape[0]
    te_num = X2.shape[0]
    Y1 = np.full((tr_num, 1), label_num, dtype=int)
    Y2 = np.full((te_num, 1), label_num, dtype=int)
    try:
        Y_train = np.concatenate((Y_train, Y1), axis=0)
    except:
        Y_train = Y1
    try:
        Y_test = np.concatenate((Y_test, Y2), axis=0)
    except:
        Y_test = Y2

print('* ----------------------------- *')
print('X_train.shape:', X_train.shape)
print('Y_train.shape:', Y_train.shape)
print('X_test.shape:', X_test.shape)
print('Y_test.shape:', Y_test.shape)
print('* ----------------------------- *')


## Put sparse features into simple neural networks ##

X_train, Y_train = shuffle(X_train, Y_train, random_state=0)
X_test, Y_test = shuffle(X_test, Y_test, random_state=0)
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
tr_labels = keras.utils.to_categorical(Y_train, num_classes=10)
te_labels = keras.utils.to_categorical(Y_test, num_classes=10)

# Train the model, iterating on the data in batches of 32 samples
print('Training Model ...')
model_mlp.fit(X_train, tr_labels, epochs=30, batch_size=32)

model_mlp.evaluate(X_test, te_labels, batch_size=32)

score = model_mlp.evaluate(X_test, te_labels, batch_size=32)
print('\nTest Score:', score[1])




## Put original video into C3D networks ##
# model_c3d = c3d_utils.get_model(summary=True)



## Evaluation ##
