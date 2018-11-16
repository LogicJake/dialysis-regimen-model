# -*- coding: utf-8 -*-
# @Author: LogicJake
# @Date:   2018-11-16 10:10:26
# @Last Modified time: 2018-11-16 11:22:27
import os
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE


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
        df = df.drop(['anti_add', 'id'], axis=1)
        df = df.dropna(subset=['mode', 'machine',
                               'anti_type', 'anti_first', 'flow'])

        # # merge output
        # df['mm'] = df['machine'].str.cat(df['mode'], sep=':')
        # df = df.drop(['machine', 'mode'], axis=1)
        # df['anti_first'] = df['anti_first'].map(lambda x: str(x))
        # df['anti'] = df['anti_type'].str.cat(df['anti_first'], sep=':')
        # df = df.drop(['anti_first', 'anti_type'], axis=1)

        # df = df[['flow', 'mm', 'anti']]
        df = df[['flow', 'machine', 'mode', 'anti_type', 'anti_first']]
        df = pd.get_dummies(
            df, columns=['machine', 'mode', 'anti_type', 'anti_first'], prefix_sep='+')
        print(df.columns.values.tolist())
        df.to_csv('transformed_dataset/data.csv', index=False, header=False)

        return df

if __name__ == '__main__':
    preprocessing = Preprocessing('../dataset/output.csv')
    preprocessing.reformat()
