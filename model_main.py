# -*- coding: utf-8 -*-
# @Author: LogicJake
# @Date:   2018-10-29 18:53:00
# @Last Modified time: 2018-11-02 11:24:56
from loss_history import LossHistory
import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Dense
from keras.layers import Input
from keras.utils import plot_model
from keras import regularizers
from sklearn.cross_validation import train_test_split
from keras.models import load_model
from keras import backend as K
from keras.optimizers import SGD


seed = 7
np.random.seed(seed)

# hyperparameters
BS = 10000
l = 0.01


def my_loss(y_true, y_pred):
    return -K.mean(y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred), axis=-1)


class MainModel(object):
    """the model of other output('flow' not included)"""

    def __init__(self, path):
        super(MainModel, self).__init__()
        self.path = path

        # the paramter about label encoder
        self.label_dict = {}
        self.label_num = {}

    def build_mm_hide(self, visible):
        hide1 = Dense(10, activation='relu',
                      kernel_initializer='normal', activity_regularizer=regularizers.l1(l))(visible)
        hide2 = Dense(10, activation='relu',
                      kernel_initializer='normal', activity_regularizer=regularizers.l1(l))(hide1)
        return hide2

    def build_anti_hide(self, visible):
        hide1 = Dense(10, activation='relu',
                      kernel_initializer='normal', activity_regularizer=regularizers.l1(l))(visible)
        hide2 = Dense(10, activation='relu',
                      kernel_initializer='normal', activity_regularizer=regularizers.l1(l))(hide1)
        return hide2

    def build_add_hide(self, visible):
        hide1 = Dense(10, activation='relu',
                      kernel_initializer='normal', activity_regularizer=regularizers.l1(l))(visible)
        hide2 = Dense(10, activation='relu',
                      kernel_initializer='normal', activity_regularizer=regularizers.l1(l))(hide1)
        return hide2

    def build(self, input_dim):
        visible = Input(shape=(input_dim, ))

        mm_hide = self.build_mm_hide(visible)
        anti_hide = self.build_anti_hide(visible)
        add_hide = self.build_add_hide(visible)

        mm_num = self.label_num['mm']
        anti_num = self.label_num['anti']

        # mm
        output_mm = Dense(mm_num, activation='softmax', kernel_initializer='normal',
                          name='output_mm')(mm_hide)
        # anti
        output_anti = Dense(anti_num, activation='softmax', kernel_initializer='normal',
                            name='output_anti')(anti_hide)

        # anti_add
        output_anti_add = Dense(1, kernel_initializer='normal',
                                name='output_anti_add')(add_hide)

        model = Model(inputs=visible,
                      outputs=[output_mm, output_anti, output_anti_add])
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
        anti_add_num = 1
        mm_num = self.label_num['mm']
        anti_num = self.label_num['anti']

        anti_add_start = 0
        mm_start = anti_add_start + anti_add_num
        anti_start = mm_start + mm_num

        Y_anti_add = dataset[:, anti_add_start:mm_start]
        Y_mm = dataset[:, mm_start:anti_start]
        Y_anti = dataset[:, anti_start:anti_start + anti_num]

        return [Y_mm, Y_anti, Y_anti_add]

    def split_train_test(self, dataframe):
        X = dataframe.ix[:, 0:38].values
        Y = dataframe.ix[:, 38:41]

        encoded_columns = ['mm', 'anti']

        # one-hot encoder
        Y = pd.get_dummies(
            Y, columns=encoded_columns, prefix_sep='+')
        column_name = Y.columns.values.tolist()
        print(column_name)
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
        dataframe = pd.read_csv(self.path)

        X_tranin, Y_tranin, X_test, Y_test = self.split_train_test(dataframe)

        model = self.build(X_tranin.shape[1])

        learning_rate = 0.1
        EPOCHS = 1000

        decay_rate = learning_rate / EPOCHS
        momentum = 0.8
        sgd = SGD(lr=learning_rate, momentum=momentum,
                  decay=decay_rate, nesterov=False)
        # Fit the model

        model.compile(loss=[my_loss, my_loss, "mean_squared_error"],
                      optimizer=sgd,
                      metrics=["accuracy"])

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
            model = load_model('model.h5')
            with open('labels' + '.txt', 'r') as f:
                a = f.read()
                label_dict = eval(a)
        else:
            model = self.model
            label_dict = self.label_dict

        X_predict = np.array([0, 70, 44.3, 43.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        Y_predict = model.predict(X_predict.reshape((-1, 38)))

        mode = self.number2label(label_dict, Y_predict[0][0], 'mode')
        machine = self.number2label(label_dict, Y_predict[1][0], 'machine')
        anti_type = self.number2label(label_dict, Y_predict[2][0], 'anti_type')
        anti_first = self.number2label(
            label_dict, Y_predict[3][0], 'anti_first')
        anti_add = Y_predict[4][0][0]

        print('mode:' + mode)
        print('machine:' + machine)
        print('anti_type:' + anti_type)
        print('anti_first:' + anti_first)
        print('anti_add:' + str(int(anti_add)))


if __name__ == '__main__':
    model = MainModel('transformed_dataset/final.csv')
    model.train()
    # model_big.predict(True)
