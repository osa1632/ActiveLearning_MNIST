'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import to_categorical


def create_model():
    model = Sequential()
    model.add(Reshape((28, 28,1), input_shape=(784,)))
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model

class CNNClassifiear_cat(KerasClassifier):
    def fit(self, x, y, **kwargs):
        num_classes = kwargs.get('num_classes',10)
        return super(CNNClassifiear_cat,self).fit(x,to_categorical(y,num_classes))

CNNClassifiear = CNNClassifiear_cat(build_fn=create_model, epochs=4, batch_size=10, verbose=0)
