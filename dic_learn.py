import os
from os import listdir
from os.path import isfile, join

import numpy as np
import scipy.io

import ksvd
from ksvd import ApproximateKSVD


dataset_name = 'weizmann'
root_dir = '../results/' + dataset_name + '_features/'
weizmann_label_dic = [ 'bend', 'jack', 'jump', 'pjump', 'run', 'side', 'skip', 'walk', 'wave1', 'wave2' ]
weizmann_label_num = len(weizmann_label_dic)


## Reading Features ##

print('Reading Features ...')
files = []
labels = []
for file in os.listdir(root_dir):
    if file.endswith(".mat"):
        files.append(os.path.join(root_dir, file))
        video_name = file.split('.')[0]
        labels.append(video_name.split('_')[-1])

# print(files)
# print(labels)

Descr = ['EMPTY'] * weizmann_label_num
# print(Descr)

for file, label in zip(files, labels):

    print(file, label)
    try:
        label_num = weizmann_label_dic.index(label)
    except:
        label_num = weizmann_label_dic.index(label[0:-1])
    print(label_num)

    descr = scipy.io.loadmat(file)['feature']
    print(descr.shape)

    try:
        Descr[label_num] = np.concatenate((Descr[label_num], descr), axis=0)
    except:
        Descr[label_num] = descr

print('Total number of videos:', len(Descr))
print(Descr[0].shape)


## Dictionary Learning:

print('Dictionary Learning ...')
n_components = 128
for X in Descr:
    X = X.T
    aksvd = ApproximateKSVD(n_components=n_components)
    dictionary = aksvd.fit(X).components_
    gamma = aksvd.transform(X)
    print('DicShape:', dictionary.shape)
    print('gammaShape:', gamma.shape)




