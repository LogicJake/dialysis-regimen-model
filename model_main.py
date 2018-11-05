# -*- coding: utf-8 -*-
# @Author: LogicJake
# @Date:   2018-10-29 18:53:00
# @Last Modified time: 2018-11-05 17:40:32
from loss_history import LossHistory
import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, Dense
from keras.utils import plot_model
from keras import regularizers, optimizers
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras import backend as K
from keras.models import Sequential


seed = 7
np.random.seed(seed)

# hyperparameters
BS = 10000
learning_rate = 0.001
EPOCHS = 10000


class MainModel(object):
    """the model of other output('flow' not included)"""

    def __init__(self, path):
        super(MainModel, self).__init__()
        self.path = path

        # the paramter about label encoder
        self.label_dict = {}
        self.label_num = {}

    def build(self, input_dim):
        mm_num = self.label_num['mm']

        model = Sequential()
        model.add(Dense(30, input_dim=input_dim,
                        kernel_initializer='normal', activation='relu'))

        model.add(Dense(25, input_dim=input_dim,
                        kernel_initializer='normal', activation='relu'))

        model.add(Dense(15, input_dim=input_dim,
                        kernel_initializer='normal', activation='relu'))

        model.add(Dense(10, input_dim=input_dim,
                        kernel_initializer='normal', activation='relu'))

        model.add(
            Dense(mm_num, kernel_initializer='random_uniform', activation='sigmoid'))

        plot_model(model, to_file='model/model.png')
        return model

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

        mm_start = 0

        Y_mm = dataset[:, mm_start:mm_start + mm_num]

        return [Y_mm]

    def split_train_test(self, dataframe):
        X = dataframe.ix[:, 0:38].values
        Y = dataframe.ix[:, 38:39]

        encoded_columns = ['mm']

        # one-hot encoder
        Y = pd.get_dummies(
            Y, columns=encoded_columns, prefix_sep='+')
        column_name = Y.columns.values.tolist()
        self.label_counter(column_name, encoded_columns)

        Y = Y.values

        trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2)

        trainY = self.split_Y(trainY)
        testY = self.split_Y(testY)

        return trainX, trainY, testX, testY

    def save_model(self, model):
        self.model = model
        model.save('model/model.h5')
        with open('model/labels' + '.txt', 'w') as outfile:
            outfile.write(str(self.label_dict))

    def train(self):
        dataframe = pd.read_csv(self.path, header=None, names=['sex', 'age', 'dweight', 'cweight', 'd_0', 'd_1', 'd_2', 'd_3', 'd_4', 'd_5', 'd_6', 'd_7', 'd_8', 'd_9', 'd_10', 'd_11', 'd_12',
                                                               'd_13', 'd_14', 'd_15', 'd_16', 'c_0', 'c_1', 'c_2', 'c_3', 'c_4', 'c_5', 'c_6', 'c_7', 'c_8', 'c_9', 'c_10', 'c_11', 'c_12', 'c_13', 'c_14', 'c_15', 'c_16', 'mm'])

        X_tranin, Y_tranin, X_test, Y_test = self.split_train_test(dataframe)

        model = self.build(X_tranin.shape[1])

        # Fit the model
        sgd = optimizers.Adam(lr=learning_rate, beta_1=0.9,
                              beta_2=0.999, epsilon=1e-08)

        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd, metrics=['accuracy'])

        history = LossHistory()

        model.fit(X_tranin, Y_tranin, batch_size=BS,
                  validation_data=(X_test, Y_test),
                  epochs=EPOCHS, verbose=1, callbacks=[history])

        self.save_model(model)

        # 绘制acc-loss曲线
        history.loss_plot('epoch')

    def number2label(self, label_dict, aa, label_name):
        mode_labels = label_dict[label_name]

        labels = aa.tolist()
        return mode_labels[labels.index(max(labels))]

    def predict(self, load):
        if load:
            model = load_model('model/model.h5')
            with open('labels' + '.txt', 'r') as f:
                a = f.read()
                label_dict = eval(a)
        else:
            model = self.model
            label_dict = self.label_dict

        X_predict = np.array([1.0, 52.0, 69.1, 69.7, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        Y_predict = model.predict(X_predict.reshape((-1, 38)))

        mm = self.number2label(label_dict, Y_predict[0][0], 'mm')

        print('mm:' + mm)


if __name__ == '__main__':
    model = MainModel('transformed_dataset/final.csv')
    model.train()
    # model.predict(True)
