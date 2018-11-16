# -*- coding: utf-8 -*-
# @Author: LogicJake
# @Date:   2018-11-16 10:10:26
# @Last Modified time: 2018-11-16 16:47:34
import os
import pandas as pd
import numpy as np


def create_dir():
    tmp_dir = 'transformed_dataset'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)


class Preprocessing(object):

    def __init__(self, data_path):
        super(Preprocessing, self).__init__()

        self.data_path = data_path
        create_dir()

    def reformat(self):
        df = pd.read_csv(self.data_path)
        df = df[['dweight', 'sex', 'age', 'disease',
                 'complication', 'cweight']]

        sex_mapping = {
            '女': 0,
            '男': 1
        }
        # change the value of sex to number
        df['sex'] = df['sex'].map(sex_mapping)

        df_drop = df[(df['cweight'] == 0)]
        df = df.drop(df_drop.index)

        df['disease'] = df['disease'].map(lambda x: np.nan if type(
            x) != float and (len(x) < 2 or x.isspace()) else x)
        df['complication'] = df['complication'].map(
            lambda x: np.nan if type(x) != float and (len(x) < 2 or x.isspace()) else x)

        # one-hot
        disease = []
        df.apply((lambda row: disease.extend(row['disease'].split(',')) if type(
            row['disease']) != float else 1), axis=1)
        disease = list(set(disease))
        disease.sort()
        for index, d in enumerate(disease):
            df['d_' + str(index)] = df.apply(
                lambda row: 1 if type(row['disease']) == str and d in row['disease'] else 0, axis=1)
        df = df.drop(['disease'], axis=1)

        complication = []
        df.apply((lambda row: complication.extend(row['complication'].split(',')) if type(
            row['complication']) != float else 1), axis=1)
        complication = list(set(complication))
        complication.sort()
        for index, c in enumerate(complication):
            df['c_' + str(index)] = df.apply(
                lambda row: 1 if type(row['complication']) == str and c in row['complication'] else 0, axis=1)

        df = df.drop(['complication'], axis=1)

        df_train = df[(df['dweight'] != 0)]
        # df_train['dweight'] = df_train['dweight'] - df_train['cweight']
        # df_train.rename(columns={'dweight': 'target'}, inplace=True)
        df_train.to_csv('transformed_dataset/data.csv',
                        index=False)


if __name__ == '__main__':
    preprocessing = Preprocessing('../dataset/input.csv')
    preprocessing.reformat()
