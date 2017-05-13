import os
from os import listdir
from os.path import isfile, join

import numpy as np
import scipy.io
from sklearn.utils import shuffle

import keras
import mlp_utils
import c3d_utils
from keras import callbacks

import time

start_time = time.time()


remote = callbacks.RemoteMonitor(root='http://localhost:9000')


dataset_name = 'weizmann'
# dataset_name = 'kth'
label_dic = [ 'bend', 'jack', 'jump', 'pjump', 'run', 'side', 'skip', 'walk', 'wave1', 'wave2' ]
label_dic_num = len(label_dic)
# label_dic = [ 'boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking' ]
# label_dic_num = len(label_dic)
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
        label_num = label_dic.index(label)
    except:
        label_num = label_dic.index(label[0:-1])

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

model_mlp = mlp_utils.get_model(summary=True, class_num=label_dic_num)
model_mlp.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# # Generate dummy data
# data = np.random.random((1000, 100))
# labels = np.random.randint(10, size=(1000, 1))

# Convert labels to categorical one-hot encoding
tr_labels = keras.utils.to_categorical(Y_train, num_classes=label_dic_num)
te_labels = keras.utils.to_categorical(Y_test, num_classes=label_dic_num)

# Train the model, iterating on the data in batches of 32 samples
print('Training Model ...')
model_mlp.fit(X_train, tr_labels, epochs=50, batch_size=256, validation_data=(X_test, te_labels), callbacks=[remote])

# model_mlp.evaluate(X_test, te_labels, batch_size=32)

pre = model_mlp.predict(X_test)
score = model_mlp.evaluate(X_test, te_labels, batch_size=32)
print('\nTest Score:', score[1])

print("--- %s seconds ---" % (time.time() - start_time))

scipy.io.savemat('../results/pre.mat', {'te_labels': te_labels, 'pre': pre})

## Put original video into C3D networks ##
# model_c3d = c3d_utils.get_model(summary=True)



## Evaluation ##
