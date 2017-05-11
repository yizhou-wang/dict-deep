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
train_files = []
train_labels = []
test_files = []
test_labels = []
for file in os.listdir(root_dir):
    if file.endswith(".mat"):
        video_name = file.split('.')[0]
        if video_name.split('_')[-1] == 'train':
            train_files.append(os.path.join(root_dir, file))
            train_labels.append(video_name.split('_')[1])
        else:
            test_files.append(os.path.join(root_dir, file))
            test_labels.append(video_name.split('_')[1])

# print(test_files)
# print(test_labels)

Descr = ['EMPTY'] * weizmann_label_num
# print(Descr)
for file, label in zip(train_files, train_labels):

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


test_Descr = ['EMPTY'] * weizmann_label_num
for file, label in zip(test_files, test_labels):

    # print(file, label)
    try:
        label_num = weizmann_label_dic.index(label)
    except:
        label_num = weizmann_label_dic.index(label[0:-1])
    # print(label_num)

    descr = scipy.io.loadmat(file)['feature']
    # print(descr.shape)

    try:
        test_Descr[label_num] = np.concatenate((test_Descr[label_num], descr), axis=0)
    except:
        test_Descr[label_num] = descr


print('Number of Classes:', len(Descr))
print('* ----------------------------- *')
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
print('* ----------------------------- *')
print('test_Descr[0].shape', test_Descr[0].shape)
print('test_Descr[1].shape', test_Descr[1].shape)
print('test_Descr[2].shape', test_Descr[2].shape)
print('test_Descr[3].shape', test_Descr[3].shape)
print('test_Descr[4].shape', test_Descr[4].shape)
print('test_Descr[5].shape', test_Descr[5].shape)
print('test_Descr[6].shape', test_Descr[6].shape)
print('test_Descr[7].shape', test_Descr[7].shape)
print('test_Descr[8].shape', test_Descr[8].shape)
print('test_Descr[9].shape', test_Descr[9].shape)
print('* ----------------------------- *')


## Dictionary Learning:

print('Dictionary Learning ...')
R = np.random.randn(1728, 256)
n_components = 256 * 2  # Over complete factor = 2
transform_n_nonzero_coefs = 20

mat_dir = '../results/' + dataset_name + '_dic_' + str(transform_n_nonzero_coefs) + '/'
if not os.path.exists(mat_dir):
    os.makedirs(mat_dir)


for label, Y1, Y2 in zip(weizmann_label_dic, Descr, test_Descr):

    print('Learning', label, '...')

    # Y subtract mean 
    mean = Y1.mean(axis=1)
    Y1 = Y1 - mean[:, np.newaxis]
    Y1 = np.dot(Y1, R)  # Y: k x 128

    mean = Y2.mean(axis=1)
    Y2 = Y2 - mean[:, np.newaxis]
    Y2 = np.dot(Y2, R)  # Y: k x 128
   

    aksvd = ApproximateKSVD(n_components=n_components, transform_n_nonzero_coefs=transform_n_nonzero_coefs)
    D1 = aksvd.fit(Y1).components_
    X1 = aksvd.transform(Y1)
    D2 = aksvd.fit(Y2).components_
    X2 = aksvd.transform(Y2)

    print('* ----------------------------- *')
    print('D.shape:', D.shape)
    print('Y1.shape:', Y1.shape)
    print('X1.shape:', X1.shape)
    print('Y2.shape:', Y2.shape)
    print('X2.shape:', X2.shape)
    print('* ----------------------------- *')

    mat_name = mat_dir + label + '.mat'
    scipy.io.savemat(mat_name, {'dic': D, 'X_train': X1, 'X_test': X2})

    print('MAT:', mat_name, 'saved!')







