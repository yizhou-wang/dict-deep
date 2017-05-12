import os
from os import listdir
from os.path import isfile, join

import numpy as np
import scipy.io

import ksvd
from ksvd import ApproximateKSVD


dataset_name = 'weizmann'
# dataset_name = 'kth'
root_dir = '../results/' + dataset_name + '_features/'
label_dic = [ 'bend', 'jack', 'jump', 'pjump', 'run', 'side', 'skip', 'walk', 'wave1', 'wave2' ]
label_dic_num = len(label_dic)
# label_dic = [ 'boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking' ]
# label_dic_num = len(label_dic)


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

print('Loading Training Data ...')
Descr = ['EMPTY'] * label_dic_num
# print(Descr)
for file, label in zip(train_files, train_labels):
    # print(file, label)
    try:
        label_num = label_dic.index(label)
    except:
        label_num = label_dic.index(label[0:-1])
    # print(label_num)

    descr = scipy.io.loadmat(file)['feature']
    # print(descr.shape)

    try:
        Descr[label_num] = np.concatenate((Descr[label_num], descr), axis=0)
    except:
        Descr[label_num] = descr

print('Loading Test Data ...')
test_Descr = ['EMPTY'] * label_dic_num
print(test_Descr)
for file, label in zip(test_files, test_labels):
    # print(file, label)
    try:
        label_num = label_dic.index(label)
    except:
        label_num = label_dic.index(label[0:-1])
    # print(label_num)

    descr = scipy.io.loadmat(file)['feature']
    # print(descr.shape)

    try:
        test_Descr[label_num] = np.concatenate((test_Descr[label_num], descr), axis=0)
    except:
        test_Descr[label_num] = descr


print('Number of Classes:', len(Descr))
print('* ----------------------------- *')
for d in Descr:
    print('Descr.shape', d.shape)
print('* ----------------------------- *')
for d in test_Descr:
    print('test_Descr.shape', d.shape)
print('* ----------------------------- *')


## Dictionary Learning:

print('Dictionary Learning ...')
R = np.random.randn(1728, 256)
n_components = 256 * 2  # Over complete factor = 2
transform_n_nonzero_coefs = 20

mat_dir = '../results/' + dataset_name + '_dic_' + str(transform_n_nonzero_coefs) + '/'
if not os.path.exists(mat_dir):
    os.makedirs(mat_dir)


for label, Y1, Y2 in zip(label_dic, Descr, test_Descr):

    print('Learning', label, '...')

    # Y subtract mean 
    mean = Y1.mean(axis=1)
    Y1 = Y1 - mean[:, np.newaxis]
    Y1 = np.dot(Y1, R)  # Y: k x 128

    mean = Y2.mean(axis=1)
    Y2 = Y2 - mean[:, np.newaxis]
    Y2 = np.dot(Y2, R)  # Y: k x 128
   

    aksvd = ApproximateKSVD(n_components=n_components, transform_n_nonzero_coefs=transform_n_nonzero_coefs)
    D = aksvd.fit(Y1).components_
    X1 = aksvd.transform(Y1)
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







