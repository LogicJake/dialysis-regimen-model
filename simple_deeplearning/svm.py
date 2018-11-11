# -*- coding: utf-8 -*-
# @Author: LogicJake
# @Date:   2018-11-05 10:53:43
# @Last Modified time: 2018-11-11 15:48:42
from sklearn import svm
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def svm_analyse():
    df = pd.read_csv('transformed_dataset/final.csv', header=None)
    X_train = df.iloc[:, 0:38].values
    Y_train = df.iloc[:, 38:39]
    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(X_train, Y_train)

    df = pd.read_csv('transformed_dataset/test.csv', header=None)
    X_test = df.iloc[:, 0:38].values
    Y_test = df.iloc[:, 40:41].values
    Y_predict = clf.predict(X_test)

    C = confusion_matrix(Y_test, Y_predict)

    print(C)

    print(classification_report(Y_test, Y_predict))


if __name__ == '__main__':
    svm_analyse()
