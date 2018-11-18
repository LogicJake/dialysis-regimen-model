# -*- coding: utf-8 -*-
# @Author: LogicJake
# @Date:   2018-10-29 18:53:24
# @Last Modified time: 2018-11-14 15:05:10
from preprocessing import Preprocessing
from model import DNNModel
import os


if __name__ == '__main__':
    preprocessing = Preprocessing(
        '../dataset/input.csv', '../dataset/output.csv')
    preprocessing.reformat()

    model = DNNModel()
    model.train('transformed_dataset/final.csv')
    model.analyse()
