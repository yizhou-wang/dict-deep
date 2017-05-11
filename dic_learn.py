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

    # print(file, label)
    try:
        label_num = weizmann_label_dic.index(label)
    except:
        label_num = weizmann_label_dic.index(label[0:-1])
    # print(label_num)

    descr = scipy.io.loadmat(file)['feature']
    # print(descr.shape)

    try:
        Descr[label_num] = np.concatenate((Descr[label_num], descr), axis=0)
    except:
        Descr[label_num] = descr


print('Number of Classes:', len(Descr))
print('Descr[0].shape', Descr[0].shape)
print('Descr[1].shape', Descr[1].shape)
print('Descr[2].shape', Descr[2].shape)
print('Descr[3].shape', Descr[3].shape)
print('Descr[4].shape', Descr[4].shape)
print('Descr[5].shape', Descr[5].shape)
print('Descr[6].shape', Descr[6].shape)
print('Descr[7].shape', Descr[7].shape)
print('Descr[8].shape', Descr[8].shape)
print('Descr[9].shape', Descr[9].shape)


## Dictionary Learning:

print('Dictionary Learning ...')
R = np.random.randn(1728, 128)
n_components = 128 * 2  # Over complete factor = 2
transform_n_nonzero_coefs = 12

for Y, label in zip(Descr, weizmann_label_dic):

    # Y subtract mean ????????

    Y = np.dot(Y, R)  # Y: k x 128

    aksvd = ApproximateKSVD(n_components=n_components, transform_n_nonzero_coefs=transform_n_nonzero_coefs)
    D = aksvd.fit(Y).components_
    X = aksvd.transform(Y)

    print('YShape:', Y.shape)
    print('DShape:', D.shape)
    print('XShape:', X.shape)

    mat_dir = '../results/' + dataset_name + '_dic/'
    if not os.path.exists(mat_dir):
        os.makedirs(mat_dir)
    mat_name = mat_dir + label + '.mat'
    scipy.io.savemat(mat_name, {'dic': D, 'X': X})

    print('MAT:', mat_name, 'saved!')




