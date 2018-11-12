# -*- coding: utf-8 -*-
# @Author: LogicJake
# @Date:   2018-11-12 09:41:22
# @Last Modified time: 2018-11-12 19:18:34
import os
import time

import pandas as pd
from keras import optimizers
from keras.engine import Model, Input
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from preprocessing import series_length

plot = True

learning_rate = 0.01
decay = 0.004
EPOCHS = 100
BS = 10000


class LSTMModel(object):

    def __init__(self, path):
        super(LSTMModel, self).__init__()
        self.path = path
        self.label_dict = {}
        self.label_num = {}
        self.id = str(int(time.time()))

    def train(self):
        df = pd.read_csv(self.path)
        X = df.iloc[:, :-2]
        Y = df.iloc[:, -2:]

        encoded_columns = ['mm(t)', 'anti(t)']
        Y = pd.get_dummies(Y, columns=encoded_columns, prefix_sep='+')
        column_name = Y.columns.values.tolist()
        self.label_counter(column_name, encoded_columns)

        # normalize features
        scaler = MinMaxScaler(feature_range=(0, 1))
        X = scaler.fit_transform(X.values)

        trainX, testX, trainY, testY = train_test_split(
            X, Y.values, test_size=0.1)

        # reshape input to be 3D [samples, timesteps, features]
        trainX = trainX.reshape(
            (trainX.shape[0], series_length, int(trainX.shape[1] / series_length)))
        testX = testX.reshape(
            (testX.shape[0], series_length, int(testX.shape[1] / series_length)))

        trainY = self.split_Y(trainY)
        testY = self.split_Y(testY)

        input_laywer = Input(
            shape=(trainX.shape[1], trainX.shape[2]), name='input_x')

        lstm = LSTM(50)(input_laywer)

        mm_num = self.label_num['mm(t)']
        anti_num = self.label_num['anti(t)']

        mm_output = Dense(
            mm_num, kernel_initializer='normal', activation='sigmoid', name='mm')(lstm)
        anti_output = Dense(
            anti_num, kernel_initializer='normal', activation='sigmoid', name='anti')(lstm)
        model = Model(inputs=input_laywer, outputs=[mm_output, anti_output])

        self.save_model(model)

        optimizer = optimizers.Adam(lr=learning_rate, beta_1=0.9,
                                    beta_2=0.999, epsilon=1e-08, decay=decay)
        model.compile(loss=['categorical_crossentropy',
                            'categorical_crossentropy'], optimizer=optimizer, metrics=['accuracy'])
        history = model.fit(trainX, trainY, epochs=EPOCHS, batch_size=BS, validation_data=(
            testX, testY), verbose=1)

        # plot history
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()

    def split_Y(self, dataset):
        mm_num = self.label_num['mm(t)']
        anti_num = self.label_num['anti(t)']

        mm_start = 0
        anti_start = mm_start + mm_num

        Y_mm = dataset[:, mm_start:anti_start]
        Y_anti = dataset[:, anti_start:anti_start + anti_num]

        return [Y_mm, Y_anti]

    def label_counter(self, columns, encoded_columns):
        for column in encoded_columns:
            index = 0
            label = {}
            for column_name in columns:
                if column_name.startswith(column + "+"):
                    label[index] = column_name.split('+')[1]
                    index += 1
            self.label_dict[column] = label
            self.label_num[column] = index

    def save_model(self, model):

        folder = 'model' + os.path.sep + self.id + os.path.sep
        if not os.path.exists(folder):
            os.makedirs(folder)

        self.model = model
        model.save(folder + 'model.h5')
        with open(folder + 'labels.txt', 'w') as outfile:
            outfile.write(str(self.label_dict))
        if plot:
            plot_model(model, to_file=folder +
                       'model.png', show_shapes=True)


if __name__ == '__main__':
    model = LSTMModel('transformed_dataset/final.csv')
    model.train()
