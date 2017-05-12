from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD

'''
dim_ordering issue:
- 'th'-style dim_ordering: [batch, channels, depth, height, width]
- 'tf'-style dim_ordering: [batch, depth, height, width, channels]
'''

def get_model(summary=False, backend='tf', class_num=10):
    """ Return the Keras model of the network
    """
    model = Sequential()
    if backend == 'tf':
        input_shape=(256 * 2,) # l, h, w, c
    else:
        input_shape=(256 * 2,) # c, l, h, w

    # FC layers group
    model.add(BatchNormalization(input_shape=input_shape, axis=-1, momentum=0.99, epsilon=0.001, center=True))
    model.add(Dense(4096, activation='relu', name='fc1'))
    model.add(Dropout(.3))
    model.add(Dense(2048, activation='relu', name='fc2'))
    model.add(Dropout(.3))
    # model.add(Dense(256, activation='relu', name='fc3'))
    # model.add(Dropout(.3))
    model.add(Dense(class_num, activation='softmax', name='softmax'))

    if summary:
        print(model.summary())

    return model


if __name__ == '__main__':
    model = get_model(summary=True)
