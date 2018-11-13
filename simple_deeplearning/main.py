# -*- coding: utf-8 -*-
# @Author: LogicJake
# @Date:   2018-10-29 18:53:24
# @Last Modified time: 2018-11-13 20:40:03
from preprocessing import Preprocessing
from model import MainModel
import os


def create_dir():
    dirs = ['model', 'result']
    for dir in dirs:
        if not os.path.exists(dir):
            os.mkdir(dir)


if __name__ == '__main__':
    create_dir()

    preprocessing = Preprocessing(
        '../dataset/input.csv', '../dataset/output.csv')
    preprocessing.reformat()

    model = MainModel('transformed_dataset/final.csv')
    model.train()
    model.analyse()
