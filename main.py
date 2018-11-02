# -*- coding: utf-8 -*-
# @Author: LogicJake
# @Date:   2018-10-29 18:53:24
# @Last Modified time: 2018-11-02 10:40:14
from preprocessing import Preprocessing
from model_main import MainModel
import os


def create_dir():
    dirs = ['temp', 'model', 'result']
    for dir in dirs:
        if not os.path.exists(dir):
            os.mkdir(dir)


if __name__ == '__main__':
    create_dir()

    preprocessing = Preprocessing('dataset/input.csv', 'dataset/output.csv')
    preprocessing.reformat()

    main_model = MainModel('transformed_dataset/final.csv')
    main_model.train()
