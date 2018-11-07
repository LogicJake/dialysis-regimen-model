# -*- coding: utf-8 -*-
# @Author: LogicJake
# @Date:   2018-10-29 18:53:00
# @Last Modified time: 2018-11-07 09:56:27
import warnings
from loss_history import LossHistory
import numpy as np
import pandas as pd
from keras.layers import Dense, BatchNormalization, Dropout
from keras.utils import plot_model
from keras import optimizers
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras import backend as K
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import os

# ignore warnings
warnings.filterwarnings('ignore')

seed = 7
np.random.seed(seed)
plot = False

# hyperparameters
BS = 10000
learning_rate = 0.01
EPOCHS = 1000
decay = 0.004


class MainModel(object):
    """the model of other output('flow' not included)"""

    def __init__(self, path):
        super(MainModel, self).__init__()
        self.path = path

        # the paramter about label encoder
        self.label_dict = {}
        self.label_num = {}

    def build(self, input_dim):
        anti_num = self.label_num['anti']

        model = Sequential()
        model.add(BatchNormalization(input_shape=(input_dim,), scale=False))
        model.add(Dense(35, input_dim=input_dim,
                        kernel_initializer='normal', activation='relu'))

        model.add(BatchNormalization())
        model.add(Dense(40, kernel_initializer='normal', activation='relu'))

        model.add(BatchNormalization())
        model.add(Dense(45, kernel_initializer='normal', activation='relu'))

        model.add(BatchNormalization())
        model.add(Dense(40, kernel_initializer='normal', activation='relu'))

        model.add(BatchNormalization())
        model.add(Dense(35, kernel_initializer='normal', activation='relu'))

        model.add(BatchNormalization())
        model.add(Dense(30, kernel_initializer='normal', activation='relu'))

        model.add(BatchNormalization())
        model.add(Dense(25, kernel_initializer='normal', activation='relu'))

        model.add(BatchNormalization())
        model.add(Dense(20, kernel_initializer='normal', activation='relu'))

        model.add(BatchNormalization())
        model.add(Dense(15, kernel_initializer='normal', activation='relu'))

        model.add(BatchNormalization())
        model.add(Dense(10, kernel_initializer='normal', activation='relu'))

        model.add(
            Dense(anti_num, kernel_initializer='normal', activation='sigmoid'))

        return model

    def train(self):
        dataframe = pd.read_csv(self.path, header=None, names=['sex', 'age', 'dweight', 'cweight', 'd_0', 'd_1', 'd_2', 'd_3', 'd_4', 'd_5', 'd_6', 'd_7', 'd_8', 'd_9', 'd_10', 'd_11', 'd_12',
                                                               'd_13', 'd_14', 'd_15', 'd_16', 'c_0', 'c_1', 'c_2', 'c_3', 'c_4', 'c_5', 'c_6', 'c_7', 'c_8', 'c_9', 'c_10', 'c_11', 'c_12', 'c_13', 'c_14', 'c_15', 'c_16', 'anti'])

        X_tranin, Y_tranin, X_val, Y_val = self.split_train_test(dataframe)

        model = self.build(X_tranin.shape[1])

        # Fit the model
        optimizer = optimizers.Adam(lr=learning_rate, beta_1=0.9,
                                    beta_2=0.999, epsilon=1e-08, decay=decay)
        # optimizer = optimizers.SGD(
        #     lr=learning_rate, momentum=0., decay=0., nesterov=False)

        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer, metrics=['accuracy'])

        history = LossHistory()

        model.fit(X_tranin, Y_tranin, batch_size=BS,
                  validation_data=(X_val, Y_val),
                  epochs=EPOCHS, verbose=1, callbacks=[history])

        self.history = history
        self.save_model(model)

    def analyse(self, load=False):
        df = pd.read_csv('transformed_dataset/test.csv', header=None)

        X_test = df.iloc[:, 0:38].values
        Y_test = df.iloc[:, 40:41].values

        Y_predict = self.predict(X_test, load)

        folder_name = 'result' + os.path.sep + \
            str(self.history.acc['epoch'][-1]) + os.path.sep
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

        cm = confusion_matrix(Y_test, Y_predict)
        np.savetxt(folder_name + 'cm.txt', cm,
                   fmt=['%s'] * cm.shape[1])

        cr = classification_report(Y_test, Y_predict)
        with open(folder_name + 'cr.txt', 'w') as outfile:
            outfile.write(cr)
        self.history.loss_plot(plot, folder_name, 'epoch')

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
        return self.number2label(label_dict, Y_predict, 'anti')

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
        mm_num = self.label_num['anti']

        mm_start = 0

        Y_mm = dataset[:, mm_start:mm_start + mm_num]

        return [Y_mm]

    def split_train_test(self, dataframe):
        X = dataframe.iloc[:, 0:38].values
        Y = dataframe.iloc[:, 38:39]

        encoded_columns = ['anti']

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

    def save_model(self, model):
        cur_acc = self.history.acc['epoch'][-1]

        folder = 'model' + os.path.sep + \
            str(cur_acc) + os.path.sep
        if not os.path.exists(folder):
            os.mkdir(folder)

        self.model = model
        model.save(folder + 'model.h5')
        with open(folder + 'labels.txt', 'w') as outfile:
            outfile.write(str(self.label_dict))
        if plot:
            plot_model(model, to_file=folder + 'model.png', show_shapes=True)

        best = True
        for file in os.listdir('model'):
            if os.path.isdir('model' + os.path.sep + file):
                if eval(file) > cur_acc:
                    best = False
        if best:
            model.save('model/model.h5')
            with open('model/labels.txt', 'w') as outfile:
                outfile.write(str(self.label_dict))


if __name__ == '__main__':
    model = MainModel('transformed_dataset/final.csv')
    model.train()
    model.analyse()
    # model.predict(True)
