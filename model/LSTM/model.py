# -*- coding: utf-8 -*-
# @Author: LogicJake
# @Date:   2018-11-12 09:41:22
# @Last Modified time: 2018-11-20 10:06:23
import os
import time

import pandas as pd
from keras import optimizers
from keras.engine import Model, Input
from keras.layers import Dense, LSTM, BatchNormalization
from keras.utils import plot_model
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import logging
import re

pwd = os.path.abspath(os.path.dirname(__file__)) + os.path.sep
td_path = pwd + 'transformed_dataset' + os.path.sep + 'final.csv'
model_dir = pwd + 'model' + os.path.sep
res_dir = pwd + 'result' + os.path.sep


PLOT = True

LR = 0.01
DECAY = 0.004
EPOCHS = 500
BS = 1000
VERBOSE = 1
UNITS = 50
RD = 0.2


def get_logger():
    logger = logging.getLogger('LSTM')
    logger.setLevel(logging.INFO)
    # create file handler
    log_path = pwd + "log.txt"
    fh = logging.FileHandler(log_path)

    # create formatter
    fmt = "%(asctime)s-%(name)s-%(levelname)s: %(message)s"
    datefmt = "%Y/%d/%m %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt)

    # add handler and formatter to logger
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


class LSTMModel(object):

    def __init__(self):
        super(LSTMModel, self).__init__()
        self.id = str(int(time.time()))
        self.label_dict = {}
        self.label_num = {}

        # make required Folder
        dirs = [model_dir, res_dir + self.id + os.path.sep +
                'mm', res_dir + self.id + os.path.sep + 'anti']
        for dir in dirs:
            if not os.path.exists(dir):
                os.makedirs(dir)

    def train(self):
        from loss_history import LossHistory
        from preprocessing import series_length
        df = pd.read_csv(td_path)

        '''
        determine the number of output based on quantity of colunms end with '(t)'
        split X(input) and Y(output)
        '''
        cols = df.columns.values.tolist()
        output_num = 0
        for column_name in cols:
            if column_name[-3:] == "(t)":
                output_num += 1
        X = df.iloc[:, :-output_num]
        Y = df.iloc[:, -output_num:]

        '''
        in the preprocessing, the output 'mm' and 'anti' are encoded by one-hot
        we get the mapping with label_counter function.
        '''
        encoded_columns = ['mm', 'anti']
        column_name = Y.columns.values.tolist()
        self.label_counter(column_name, encoded_columns)

        X = X.values
        Y = Y.values
        trainX, testX, trainY, testY = train_test_split(
            X, Y, test_size=0.1)

        # reshape input to be 3D [samples, timesteps, features]
        trainX = trainX.reshape(
            (trainX.shape[0], series_length, int(trainX.shape[1] / series_length)))
        testX = testX.reshape(
            (testX.shape[0], series_length, int(testX.shape[1] / series_length)))

        trainY = self.split_Y(trainY)
        testY = self.split_Y(testY)

        input_laywer = Input(
            shape=(trainX.shape[1], trainX.shape[2]), name='input_x')

        # BatchNormalization is best :)
        bn = BatchNormalization()(input_laywer)
        lstm1 = LSTM(units=UNITS, recurrent_dropout=RD)(bn)
        lstm2 = LSTM(units=UNITS, recurrent_dropout=RD)(bn)

        mm_num = self.label_num['mm']
        anti_num = self.label_num['anti']

        mm_output = Dense(
            mm_num, kernel_initializer='normal', activation='sigmoid', name='mm')(lstm1)
        anti_output = Dense(
            anti_num, kernel_initializer='normal', activation='sigmoid', name='anti')(lstm2)
        model = Model(inputs=input_laywer, outputs=[mm_output, anti_output])

        plot_model(model, 'model.png', show_shapes=True)
        # optimizer = optimizers.Adam(lr=LR, beta_1=0.9,
        #                             beta_2=0.999, epsilon=1e-08, decay=DECAY)
        # model.compile(loss=['categorical_crossentropy',
        #                     'categorical_crossentropy'], optimizer=optimizer, metrics=['accuracy'])

        # history = LossHistory()
        # model.fit(trainX, trainY, epochs=EPOCHS, batch_size=BS, validation_data=(
        #     testX, testY), verbose=VERBOSE,  callbacks=[history])

        # history.save_loss(PLOT, res_dir +
        #                   self.id + os.path.sep)
        # self.model = model
        # self.history = history

        # mm_acc = history.mm_val_acc[-1]
        # anti_acc = history.anti_val_acc[-1]
        # self.save_model(model, mm_acc, anti_acc)

        # logger = get_logger()
        # logger.info(self.id + ': training over. mm acc: {' + str(mm_acc) + '}\tanti acc: {' + str(anti_acc) +
        #             '}\thyperparameters are ' + str(LR) + '\t' +
        # str(DECAY) + '\t' + str(EPOCHS) + '\t' + str(BS) + '\t' +
        # str(series_length))

    def split_Y(self, dataset):
        # split the output to two ndarray and put them in the array
        mm_num = self.label_num['mm']
        anti_num = self.label_num['anti']

        mm_start = 0
        anti_start = mm_start + mm_num

        Y_mm = dataset[:, mm_start:anti_start]
        Y_anti = dataset[:, anti_start:anti_start + anti_num]

        return [Y_mm, Y_anti]

    def predict(self, X, load=True):
        if load:
            model = load_model(model_dir + 'model.h5')
            with open(model_dir + 'labels.txt', 'r') as f:
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
        '''
        in the preprocessing, the encoed columns' name is named by
        putting column's name and its values together, connect with '+'
        '''
        for column in encoded_columns:
            index = 0
            label = {}
            for column_name in columns:
                if column_name.startswith(column + "?"):
                    label[index] = column_name.split('?')[1][:-3]
                    index += 1
            # save the correspondence between names and indexes
            self.label_dict[column] = label
            self.label_num[column] = index

    def save_model(self, model, c_mm_acc, c_anti_acc):
        '''
        if this mode's performance is better than existing model, save it
        '''
        is_better = False

        try:
            with open(pwd + 'log.txt', 'r') as f:
                lines = f.readlines()
                if len(lines) == 0:
                    is_better = True
                else:
                    last_log = lines[-1]
        except FileNotFoundError:
            is_better = True

        if not is_better:
            try:
                pattern = re.compile(r'[{](.*?)[}]', re.S)
                mm_acc, anti_acc = re.findall(pattern, last_log)
                if float(mm_acc) + float(anti_acc) < c_mm_acc + c_anti_acc:
                    is_better = True
            except ValueError:
                is_better = True

        if is_better:
            model.save(model_dir + 'model.h5')
            with open(model_dir + 'labels.txt', 'w') as outfile:
                outfile.write(str(self.label_dict))
                if PLOT:
                    plot_model(model, to_file=model_dir +
                               'model.png', show_shapes=True)


if __name__ == '__main__':
    model = LSTMModel()
    model.train()
