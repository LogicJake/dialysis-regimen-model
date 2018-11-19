# -*- coding: utf-8 -*-
# @Author: LogicJake
# @Date:   2018-11-16 10:10:15
# @Last Modified time: 2018-11-19 10:45:13
from sklearn import svm
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import os
import numpy as np
import time

pwd = os.path.abspath(os.path.dirname(__file__))
td_path = pwd + os.path.sep + 'transformed_dataset' + os.path.sep + 'data.csv'
model_dir = pwd + os.path.sep + 'model'
res_dir = pwd + os.path.sep + 'result'


class Model():

    def __init__(self):
        super(Model, self).__init__()
        self.id = str(int(time.time()))

    def train(self):
        df = pd.read_csv(td_path)

        Y = df.iloc[:, 0:1].values
        X = df.iloc[:, 1:].values

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.3)

        model = svm.SVC(kernel='rbf', C=100, gamma=0.001)

        model.fit(X_train, Y_train.ravel())
        Y_predict = model.predict(X_test)

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        joblib.dump(model, model_dir + os.path.sep + 'model.m')

        global res_dir
        res_dir = res_dir + os.path.sep + self.id

        if not os.path.exists(res_dir):
            os.makedirs(res_dir)

        cm = confusion_matrix(Y_test, Y_predict)
        np.savetxt(res_dir + os.path.sep + 'cm.txt', cm,
                   fmt=['%s'] * cm.shape[1])
        cr = classification_report(Y_test, Y_predict)
        with open(res_dir + os.path.sep + 'cr.txt', 'w') as outfile:
            outfile.write(cr)

    def predict(self, X):
        model = joblib.load(model_dir + os.path.sep + 'model.m')
        Y = model.predict(X)
        return Y


if __name__ == '__main__':
    model = Model()
    model.train()
