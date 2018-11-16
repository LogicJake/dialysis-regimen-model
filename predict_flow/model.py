# -*- coding: utf-8 -*-
# @Author: LogicJake
# @Date:   2018-11-16 10:10:15
# @Last Modified time: 2018-11-16 11:30:33
from sklearn import svm
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import os


def model():
    df = pd.read_csv('transformed_dataset/data.csv', header=None)

    Y = df.iloc[:, 0:1].values
    X = df.iloc[:, 1:].values

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.3)

    model = svm.SVC(kernel='rbf', probability=True)
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10,
                        100, 1000], 'gamma': [0.001, 0.0001]}
    grid_search = GridSearchCV(model, param_grid, n_jobs=8, verbose=0)

    grid_search.fit(X_train, Y_train)
    best_parameters = grid_search.best_estimator_.get_params()

    model = svm.SVC(kernel='rbf', C=best_parameters[
        'C'], gamma=best_parameters['gamma'], probability=True)

    model_dir = 'model'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    joblib.dump(model, model_dir + os.path.sep + 'train_model.m')
    model.fit(X_train, Y_train)
    Y_predict = model.predict(X_test)

    cm = confusion_matrix(Y_test, Y_predict)
    cr = classification_report(Y_test, Y_predict)
    print(cm)
    print(cr)


if __name__ == '__main__':
    model()
