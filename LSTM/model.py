# -*- coding: utf-8 -*-
# @Author: LogicJake
# @Date:   2018-11-12 09:41:22
# @Last Modified time: 2018-11-13 21:47:59
import os
import time

import pandas as pd
from keras import optimizers
from keras.engine import Model, Input
from keras.layers import Dense, LSTM, BatchNormalization
import matplotlib.pyplot as plt
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from preprocessing import series_length
import logging
from keras.models import load_model
import re
from loss_history import LossHistory


logging.basicConfig(filename='log.txt', level=logging.ERROR)

plot = False

learning_rate = 0.01
decay = 0.004
EPOCHS = 1000
BS = 10000
verbose = 1


class LSTMModel(object):

    def __init__(self, path):
        super(LSTMModel, self).__init__()
        self.path = path
        self.label_dict = {}
        self.label_num = {}
        self.id = str(int(time.time()))

    def train(self):
        df = pd.read_csv(self.path)

        cols = df.columns.values.tolist()
        output_num = 0
        for column_name in cols:
            if column_name[-3:] == "(t)":
                output_num += 1
        X = df.iloc[:, :-output_num]
        Y = df.iloc[:, -output_num:]

        encoded_columns = ['mm', 'anti']
        # Y = pd.get_dummies(Y, columns=encoded_columns, prefix_sep='+')
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
        bn = BatchNormalization()(input_laywer)
        lstm1 = LSTM(50)(bn)
        lstm2 = LSTM(50)(bn)

        mm_num = self.label_num['mm']
        anti_num = self.label_num['anti']

        mm_output = Dense(
            mm_num, kernel_initializer='normal', activation='sigmoid', name='mm')(lstm1)
        anti_output = Dense(
            anti_num, kernel_initializer='normal', activation='sigmoid', name='anti')(lstm2)
        model = Model(inputs=input_laywer, outputs=[mm_output, anti_output])

        optimizer = optimizers.Adam(lr=learning_rate, beta_1=0.9,
                                    beta_2=0.999, epsilon=1e-08, decay=decay)
        model.compile(loss=['categorical_crossentropy',
                            'categorical_crossentropy'], optimizer=optimizer, metrics=['accuracy'])

        history = LossHistory()
        model.fit(trainX, trainY, epochs=EPOCHS, batch_size=BS, validation_data=(
            testX, testY), verbose=verbose,  callbacks=[history])

        self.model = model
        self.id = str(int(time.time()))
        self.history = history
        self.mm_acc = history.mm_val_acc[-1]
        self.anti_acc = history.anti_val_acc[-1]

        self.save_model(model)
        self.history.save_loss(plot, 'result' + os.path.sep +
                               self.id + os.path.sep)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        logger.info(self.id + ': training over. mm acc: {' + str(self.mm_acc) + '}\tanti acc: {' + str(self.anti_acc) +
                    '}\thyperparameters are ' + str(learning_rate) + '\t' +
                    str(decay) + '\t' + str(EPOCHS) + '\t' + str(BS) + '\t' + str(series_length))

    def split_Y(self, dataset):
        mm_num = self.label_num['mm']
        anti_num = self.label_num['anti']

        mm_start = 0
        anti_start = mm_start + mm_num

        Y_mm = dataset[:, mm_start:anti_start]
        Y_anti = dataset[:, anti_start:anti_start + anti_num]

        return [Y_mm, Y_anti]

    def predict(self, X, load):
        if load:
            model = load_model('model/model.h5')
            with open('model/labels.txt', 'r') as f:
                a = f.read()
                label_dict = eval(a)
        else:
            model = self.model
            label_dict = self.label_dict

        Y_predict = model.predict(X)
        mm_predict = Y_predict[0]
        anti_predict = Y_predict[1]
        return self.number2label(label_dict, mm_predict, 'mm'), self.number2label(label_dict, anti_predict, 'anti')

    def number2label(self, label_dict, aa, label_name):
        mode_labels = label_dict[label_name]

        res = []
        for row in aa:
            labels = row.tolist()
            res.append(mode_labels[labels.index(max(labels))])

        return res

    def label_counter(self, columns, encoded_columns):
        for column in encoded_columns:
            index = 0
            label = {}
            for column_name in columns:
                if column_name.startswith(column + "+"):
                    label[index] = column_name.split('+')[1][:-3]
                    index += 1
            self.label_dict[column] = label
            self.label_num[column] = index

    def save_model(self, model):
        is_better = False

        with open('log.txt', 'r') as f:
            lines = f.readlines()
            if len(lines) == 0:
                is_better = True
            else:
                last_log = lines[-1]

        pattern = re.compile(r'[{](.*?)[}]', re.S)

        if not is_better:
            try:
                mm_acc, anti_acc = re.findall(pattern, last_log)
                if float(mm_acc) + float(anti_acc) < self.mm_acc + self.anti_acc:
                    is_better = True
            except ValueError:
                is_better = True

        if is_better:
            folder = 'model' + os.path.sep
            if not os.path.exists(folder):
                os.makedirs(folder)

            model.save(folder + 'model.h5')
            with open(folder + 'labels.txt', 'w') as outfile:
                outfile.write(str(self.label_dict))
            if plot:
                plot_model(model, to_file=folder +
                           'model.png', show_shapes=True)


if __name__ == '__main__':
    model = LSTMModel('transformed_dataset/final.csv')
    model.train()
