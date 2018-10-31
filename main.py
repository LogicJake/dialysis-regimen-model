# -*- coding: utf-8 -*-
# @Author: LogicJake
# @Date:   2018-10-29 18:53:24
# @Last Modified time: 2018-10-31 17:30:53
import preprocessing
import model_main
import os


def create_dir():
    dirs = ['temp', 'model', 'result']
    for dir in dirs:
        if not os.path.exists(dir):
            os.mkdir(dir)


if __name__ == '__main__':
    create_dir()

    # preprocessing = Preprocessing('dataset/input.csv', 'dataset/output.csv')
    # preprocessing.reformat()

    main_model = model_main.MainModel('transformed_dataset/final.csv')
    main_model.train()
