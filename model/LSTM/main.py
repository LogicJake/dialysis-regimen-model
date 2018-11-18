# -*- coding: utf-8 -*-
# @Author: LogicJake
# @Date:   2018-11-13 19:02:55
# @Last Modified time: 2018-11-18 20:46:42
from preprocessing import Preprocessing
from model import LSTMModel
if __name__ == '__main__':
    preprocessing = Preprocessing(
        '../..//dataset/input.csv', '../../dataset/output.csv')
    preprocessing.reformat()

    model = LSTMModel()
    model.train()
