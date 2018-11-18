# -*- coding: utf-8 -*-
# @Author: LogicJake
# @Date:   2018-11-16 10:10:26
# @Last Modified time: 2018-11-18 20:53:58
import os
import pandas as pd

pwd = os.path.abspath(os.path.dirname(__file__)) + os.path.sep
td_path = pwd + 'transformed_dataset' + os.path.sep
model_path = pwd + 'model' + os.path.sep


class Preprocessing(object):

    def __init__(self, data_path):
        super(Preprocessing, self).__init__()

        self.data_path = data_path
        if not os.path.exists(td_path):
            os.makedirs(td_path)

    def one_hot(self, df, name):
        names = set()
        df.apply((lambda row: names.add(str(row[name]))), axis=1)
        names = list(names)
        names.sort()

        for index, d in enumerate(names):
            df[name + '+' + str(index)] = df.apply(
                lambda row: 1 if d == row[name] else 0, axis=1)
        df = df.drop([name], axis=1)
        return df, names

    def reformat(self):
        df = pd.read_csv(self.data_path)
        df = df.drop(['anti_add', 'id'], axis=1)
        df = df.dropna(subset=['mode', 'machine',
                               'anti_type', 'anti_first', 'flow'])

        df = df[['flow', 'machine', 'mode', 'anti_type', 'anti_first']]
        df, machines = self.one_hot(df, 'machine')
        df, modes = self.one_hot(df, 'mode')
        df, anti_types = self.one_hot(df, 'anti_type')
        df, anti_firsts = self.one_hot(df, 'anti_first')

        content = {}
        content['machine'] = machines
        content['mode'] = modes
        content['anti_type'] = anti_types
        content['anti_first'] = anti_firsts

        if not os.path.exists(model_path):
            os.makedirs(model_path)
        with open(model_path + 'mapping.txt', 'w') as f:
            f.write(str(content))

        # df = pd.get_dummies(
        # df, columns=['machine', 'mode', 'anti_type', 'anti_first'],
        # prefix_sep='+')
        df.to_csv(td_path + 'data.csv', index=False)

if __name__ == '__main__':
    preprocessing = Preprocessing('../../dataset/output.csv')
    preprocessing.reformat()
