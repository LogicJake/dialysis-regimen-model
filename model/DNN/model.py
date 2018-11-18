# -*- coding: utf-8 -*-
# @Author: LogicJake
# @Date:   2018-10-29 18:53:00
# @Last Modified time: 2018-11-18 20:45:44
from keras.engine import Model, Input
import numpy as np
import pandas as pd
from keras.layers import Dense, BatchNormalization
from keras.utils import plot_model
from keras import optimizers
from sklearn.model_selection import train_test_split
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import h5py
import os
import time
import logging
import re

pwd = os.path.abspath(os.path.dirname(__file__)) + os.path.sep
td_path = pwd + 'transformed_dataset' + os.path.sep + 'data.csv'
model_dir = pwd + 'model'
res_dir = pwd + 'result'


# seed = 7
# np.random.seed(seed)

PLOT = True  # weather plot acc or loss
VERBOSE = True  # weather print info during training

# hyperparameters
BS = 2000
LR = 0.01
EPOCHS = 1000
DECAY = 0.004


def get_logger():
    logger = logging.getLogger('DNN')
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


class DNNModel(object):
    """the model of other output('flow' not included)"""

    def __init__(self):
        super(DNNModel, self).__init__()
        self.id = str(int(time.time()))
        # the paramter about label encoder
        self.label_dict = {}
        self.label_num = {}

        dirs = [model_dir, res_dir,
                res_dir + os.path.sep + self.id + os.path.sep + 'mm',
                res_dir + os.path.sep + self.id + os.path.sep + 'anti']
        for dir in dirs:
            if not os.path.exists(dir):
                os.makedirs(dir)

    def build_hide_mm(self, input):
        mm_num = self.label_num['mm']
        bn1 = BatchNormalization()(input)

        hide1 = Dense(35, kernel_initializer='normal', activation='relu')(bn1)

        bn2 = BatchNormalization()(hide1)

        hide2 = Dense(40, kernel_initializer='normal', activation='relu')(bn2)

        bn3 = BatchNormalization()(hide2)

        hide3 = Dense(45, kernel_initializer='normal', activation='relu')(bn3)

        bn4 = BatchNormalization()(hide3)

        hide4 = Dense(40, kernel_initializer='normal', activation='relu')(bn4)

        bn5 = BatchNormalization()(hide4)

        hide5 = Dense(35, kernel_initializer='normal', activation='relu')(bn5)

        bn6 = BatchNormalization()(hide5)

        hide6 = Dense(30, kernel_initializer='normal', activation='relu')(bn6)

        bn7 = BatchNormalization()(hide6)

        hide7 = Dense(25, kernel_initializer='normal', activation='relu')(bn7)

        bn8 = BatchNormalization()(hide7)

        hide8 = Dense(20, kernel_initializer='normal', activation='relu')(bn8)

        bn9 = BatchNormalization()(hide8)

        hide9 = Dense(15, kernel_initializer='normal', activation='relu')(bn9)

        bn10 = BatchNormalization()(hide9)

        hide10 = Dense(10, kernel_initializer='normal',
                       activation='relu')(bn10)

        mm_output = Dense(
            mm_num, kernel_initializer='normal', activation='sigmoid', name='mm')(hide10)

        return mm_output

    def build_hide_anti(self, input):
        anti_num = self.label_num['anti']
        bn1 = BatchNormalization()(input)

        hide1 = Dense(35, kernel_initializer='normal', activation='relu')(bn1)

        bn2 = BatchNormalization()(hide1)

        hide2 = Dense(40, kernel_initializer='normal', activation='relu')(bn2)

        bn3 = BatchNormalization()(hide2)

        hide3 = Dense(45, kernel_initializer='normal', activation='relu')(bn3)

        bn4 = BatchNormalization()(hide3)

        hide4 = Dense(40, kernel_initializer='normal', activation='relu')(bn4)

        bn5 = BatchNormalization()(hide4)

        hide5 = Dense(35, kernel_initializer='normal', activation='relu')(bn5)

        bn6 = BatchNormalization()(hide5)

        hide6 = Dense(30, kernel_initializer='normal', activation='relu')(bn6)

        bn7 = BatchNormalization()(hide6)

        hide7 = Dense(25, kernel_initializer='normal', activation='relu')(bn7)

        bn8 = BatchNormalization()(hide7)

        hide8 = Dense(20, kernel_initializer='normal', activation='relu')(bn8)

        bn9 = BatchNormalization()(hide8)

        hide9 = Dense(15, kernel_initializer='normal', activation='relu')(bn9)

        bn10 = BatchNormalization()(hide9)

        hide10 = Dense(10, kernel_initializer='normal',
                       activation='relu')(bn10)

        anti_output = Dense(
            anti_num, kernel_initializer='normal', activation='sigmoid', name='anti')(hide10)

        return anti_output

    def build(self, input_dim):
        input_laywer = Input(shape=(input_dim, ))

        mm_output = self.build_hide_mm(input_laywer)
        anti_output = self.build_hide_anti(input_laywer)

        model = Model(inputs=input_laywer, outputs=[mm_output, anti_output])
        return model

    def train(self, path):
        from loss_history import LossHistory
        self.path = path
        dataframe = pd.read_csv(self.path, header=None, names=['sex', 'age', 'dweight', 'cweight', 'd_0', 'd_1', 'd_2', 'd_3', 'd_4', 'd_5', 'd_6', 'd_7', 'd_8', 'd_9', 'd_10', 'd_11', 'd_12',
                                                               'd_13', 'd_14', 'd_15', 'd_16', 'c_0', 'c_1', 'c_2', 'c_3', 'c_4', 'c_5', 'c_6', 'c_7', 'c_8', 'c_9', 'c_10', 'c_11', 'c_12', 'c_13', 'c_14', 'c_15', 'c_16', 'mm', 'anti'])

        X_tranin, Y_tranin, X_val, Y_val = self.split_train_test(dataframe)

        model = self.build(X_tranin.shape[1])
        # Fit the model
        optimizer = optimizers.Adam(lr=LR, beta_1=0.9,
                                    beta_2=0.999, epsilon=1e-08, decay=DECAY)

        model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy'],
                      optimizer=optimizer, metrics=['accuracy'])

        history = LossHistory()

        model.fit(X_tranin, Y_tranin, batch_size=BS,
                  validation_data=(X_val, Y_val),
                  epochs=EPOCHS, verbose=VERBOSE, callbacks=[history])

        history.save_loss(PLOT, 'result' + os.path.sep +
                          self.id + os.path.sep)
        self.model = model
        self.history = history

        mm_acc = history.mm_val_acc[-1]
        anti_acc = history.anti_val_acc[-1]
        self.save_model(model, mm_acc, anti_acc)

        logger = get_logger()
        logger.info(self.id + ': training over. mm acc: {' + str(mm_acc) + '}\tanti acc: {' + str(anti_acc) +
                    '}\thyperparameters are ' + str(LR) + '\t' +
                    str(DECAY) + '\t' + str(EPOCHS) + '\t' + str(BS))

    def analyse(self, load=False):
        df = pd.read_csv('transformed_dataset/test.csv', header=None)

        X_test = df.iloc[:, 0:38].values
        mm_test = df.iloc[:, 38:39].values
        anti_test = df.iloc[:, 39:40].values

        mm_predict, anti_predict = self.predict(X_test, load)

        self.save_analyse_res('mm', mm_test, mm_predict)
        self.save_analyse_res('anti', anti_test, anti_predict)

    def save_analyse_res(self, output_name, Y_test, Y_predict):
        folder_name = 'result' + os.path.sep + self.id + \
            os.path.sep + output_name + os.path.sep

        cm = confusion_matrix(Y_test, Y_predict)
        np.savetxt(folder_name + 'cm.txt', cm,
                   fmt=['%s'] * cm.shape[1])

        cr = classification_report(Y_test, Y_predict)
        with open(folder_name + 'cr.txt', 'w') as outfile:
            outfile.write(cr)

    def predict(self, X, load=True):
        if load:
            with h5py.File(model_dir + os.path.sep + 'model.h5', 'a') as f:
                if 'optimizer_weights' in f.keys():
                    del f['optimizer_weights']

            model = load_model(model_dir + os.path.sep + 'model.h5')
            with open(model_dir + os.path.sep + 'labels.txt', 'r') as f:
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
                    label[index] = column_name.split('+')[1]
                    index += 1
            self.label_dict[column] = label
            self.label_num[column] = index

    def split_Y(self, dataset):
        mm_num = self.label_num['mm']
        anti_num = self.label_num['anti']

        mm_start = 0
        anti_start = mm_start + mm_num

        Y_mm = dataset[:, mm_start:anti_start]
        Y_anti = dataset[:, anti_start:anti_start + anti_num]

        return [Y_mm, Y_anti]

    def split_train_test(self, dataframe):
        X = dataframe.iloc[:, 0:38].values
        Y = dataframe.iloc[:, 38:40]

        encoded_columns = ['mm', 'anti']

        # one-hot encoder
        Y = pd.get_dummies(
            Y, columns=encoded_columns, prefix_sep='+')
        column_name = Y.columns.values.tolist()
        self.label_counter(column_name, encoded_columns)

        Y = Y.values

        trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.1)

        trainY = self.split_Y(trainY)
        testY = self.split_Y(testY)

        return trainX, trainY, testX, testY

    def save_model(self, model, c_mm_acc, c_anti_acc):
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

        pattern = re.compile(r'[{](.*?)[}]', re.S)

        if not is_better:
            try:
                mm_acc, anti_acc = re.findall(pattern, last_log)
                if float(mm_acc) + float(anti_acc) < c_mm_acc + c_anti_acc:
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
            if PLOT:
                plot_model(model, to_file=folder +
                           'model.png', show_shapes=True)


if __name__ == '__main__':
    model = DNNModel()
    model.train('transformed_dataset/final.csv')
    model.analyse()
