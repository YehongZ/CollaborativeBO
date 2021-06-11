
import pickle
import numpy as np
# import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.compat.v1 as tf
tf.logging.set_verbosity(tf.logging.ERROR)

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# config.gpu_options.allow_growth = True

import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers
from keras.datasets import mnist
from keras import backend as K
from sklearn.model_selection import train_test_split

class MultiLRModel():
    def __init__(self, file_name, agentList):

        self.agentList = agentList
        self.data = pickle.load(open(file_name, "rb"))

    def fit_model(self, param, s):

        config = tf.ConfigProto()
        tf.disable_v2_behavior()
        tf.reset_default_graph()
        tf.set_random_seed(s)
        np.random.seed(s)

        batch_size = np.rint(param[0]).astype(int)
        C = param[1].item()
        learning_rate = param[2]

        X_train, X_test, Y_train, Y_test = self.data[s][0], self.data[s][1], self.data[s][2], self.data[s][3]

        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        n_ft = X_train.shape[1]
        n_classes = Y_train.shape[1]

        model = Sequential()
        model.add(Flatten())
        model.add(Dense(n_classes, kernel_regularizer=regularizers.l2(C)))
        model.add(Activation('softmax'))

        opt = keras.optimizers.SGD(lr=learning_rate)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])

        history = model.fit(X_train, Y_train,
                      batch_size=batch_size,
                      epochs=5,
                      validation_data=(X_test, Y_test),
                      shuffle=True, verbose=0)

        val_acc = max(history.history['val_acc'])

        return val_acc


class MultiCNNModel():
    def __init__(self, file_name, agentList):

        self.agentList = agentList
        self.data = pickle.load(open(file_name, "rb"))

    def fit_model(self, param, s):

        config = tf.ConfigProto()
        tf.disable_v2_behavior()
        tf.reset_default_graph()
        tf.set_random_seed(s)
        np.random.seed(s)

        learning_rate = param[0]
        learning_rate_decay = param[1]
        l2_regular = param[2].item()

        num_classes = 10
        epochs = 20

        X_train, X_test, Y_train, Y_test = self.data[s][0], self.data[s][1], self.data[s][2], self.data[s][3]

        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train = 1 - X_train
        X_test = 1 - X_test

        dropout_rate = 0.0

        batch_size = 32
        conv_filters = 16
        dense_units = 8

        num_conv_layers = 2
        kernel_size = 3
        pool_size = 3

        # build the CNN model using Keras
        model = Sequential()
        model.add(Conv2D(conv_filters, (kernel_size, kernel_size), padding='same',
                         input_shape=X_train.shape[1:], kernel_regularizer=regularizers.l2(l2_regular)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
        model.add(Dropout(dropout_rate))
        model.add(Flatten())
        model.add(Dense(dense_units, kernel_regularizer=regularizers.l2(l2_regular)))
        model.add(Activation('relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))

        opt = keras.optimizers.RMSprop(lr=learning_rate, decay=learning_rate_decay)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])
        #
        history = model.fit(X_train, Y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      validation_data=(X_test, Y_test),
                      shuffle=True, verbose=0)
        val_acc = max(history.history['val_acc'])

        return val_acc
