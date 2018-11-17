# -*- coding: utf-8 -*-
# @Author: LogicJake
# @Date:   2018-11-16 10:10:26
# @Last Modified time: 2018-11-17 15:26:23
import os
import pandas as pd
import numpy as np

pwd = os.path.abspath(os.path.dirname(__file__))
td_path = pwd + os.path.sep + 'transformed_dataset' + os.path.sep


class Preprocessing(object):

    def __init__(self, data_path):
        super(Preprocessing, self).__init__()
        self.data_path = data_path

        if not os.path.exists(td_path):
            os.makedirs(td_path)

    def read_mapping(self):
        disease = None
        complication = None

        with open('../feature_mapping.txt', 'r') as f:
            content = f.readline()
            content = eval(content)
            disease = content['disease']
            complication = content['complication']

        return disease, complication

    def reformat(self):
        disease, complication = self.read_mapping()

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
        for index, d in enumerate(disease):
            df['d_' + str(index)] = df.apply(
                lambda row: 1 if type(row['disease']) == str and d in row['disease'] else 0, axis=1)
        df = df.drop(['disease'], axis=1)

        for index, c in enumerate(complication):
            df['c_' + str(index)] = df.apply(
                lambda row: 1 if type(row['complication']) == str and c in row['complication'] else 0, axis=1)

        df = df.drop(['complication'], axis=1)

        df_train = df[(df['dweight'] != 0)]
        df_train['dweight'] = df_train['dweight'] - df_train['cweight']
        df_train.rename(columns={'dweight': 'target'}, inplace=True)
        df_train.to_csv(td_path + 'data.csv', index=False)


if __name__ == '__main__':
    preprocessing = Preprocessing('../dataset/input.csv')
    preprocessing.reformat()
