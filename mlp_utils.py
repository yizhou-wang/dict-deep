from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.optimizers import SGD

'''
dim_ordering issue:
- 'th'-style dim_ordering: [batch, channels, depth, height, width]
- 'tf'-style dim_ordering: [batch, depth, height, width, channels]
'''

def get_model(summary=False, backend='tf'):
    """ Return the Keras model of the network
    """
    model = Sequential()
    if backend == 'tf':
        input_shape=(16, 112, 112) # l, h, w, c
    else:
        input_shape=(16, 112, 112) # c, l, h, w

    # FC layers group
    model.add(Dense(4096, input_shape=input_shape, activation='relu', name='fc1'))
    model.add(Dropout(.5))
    model.add(Dense(4096, activation='relu', name='fc2'))
    model.add(Dropout(.5))
    model.add(Dense(487, activation='softmax', name='fc3'))

    if summary:
        print(model.summary())

    return model


if __name__ == '__main__':
    model = get_model(summary=True)
