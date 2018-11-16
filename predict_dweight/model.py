# -*- coding: utf-8 -*-
# @Author: LogicJake
# @Date:   2018-11-16 15:06:04
# @Last Modified time: 2018-11-16 16:51:14
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import os
import numpy as np
import logging

logging.basicConfig(filename='log.txt', level=logging.ERROR)

model_dir = 'model'


def baseline():
    df = pd.read_csv('transformed_dataset/data.csv')

    Y = df.iloc[:, 0:1].values
    X = df.iloc[:, 1:].values

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.3, random_state=777)

    cweight_sum = np.sum(X_train, axis=0)[2]
    dweight_sum = np.sum(Y_train, axis=0)[0]
    aver_min = (dweight_sum - cweight_sum) / X_train.shape[0]

    Y_predict = X_test[:, 2] + aver_min
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logging.info('baseline----------->The mean squared error is %f\tThe mean absolute error is %f',
                 mean_squared_error(Y_test, Y_predict),
                 mean_absolute_error(Y_test, Y_predict))


class model():

    def train(self):
        df = pd.read_csv('transformed_dataset/data.csv')

        Y = df.iloc[:, 0:1].values
        X = df.iloc[:, 1:].values

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.3, random_state=777)

        model = svm.SVR(kernel='rbf')
        model.fit(X_train, Y_train)

        Y_predict = model.predict(X_test)

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logging.info('The mean squared error is %f\tThe mean absolute error is %f',
                     mean_squared_error(Y_test, Y_predict),
                     mean_absolute_error(Y_test, Y_predict)
                     )

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        joblib.dump(model, model_dir + os.path.sep + 'model.m')

    def predict(self, X):
        model = joblib.load(model_dir + os.path.sep + 'model.m')
        Y = model.predict(X)
        return Y

if __name__ == '__main__':
    model = model()
    model.train()
    # model.predict()
