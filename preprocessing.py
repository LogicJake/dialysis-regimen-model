# -*- coding: utf-8 -*-
# @Time    : 18-10-27 上午10:31
# @Author  : LogicJake
# @File    : preprocessing.py
import os
import pandas as pd
import numpy as np


def create_dir():
    tmp_dir = 'transformed_dataset'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)


class Preprocessing(object):

    def __init__(self, input_path, output_path):
        super(Preprocessing, self).__init__()

        self.input_path = input_path
        self.output_path = output_path
        create_dir()

    def reformat_input(self):
        df = pd.read_csv(self.input_path)
        df = df[['id', 'sex', 'age', 'disease',
                 'complication', 'dweight', 'cweight']]

        sex_mapping = {
            '女': 0,
            '男': 1
        }

        # change the value of sex to number
        df['sex'] = df['sex'].map(sex_mapping)

        # change the blank value of disease and complication to nan
        df['disease'] = df['disease'].map(lambda x: np.nan if type(
            x) != float and (len(x) < 2 or x.isspace()) else x)
        df['complication'] = df['complication'].map(
            lambda x: np.nan if type(x) != float and (len(x) < 2 or x.isspace()) else x)

        # delete that the value of cweight and dweight are all 0
        df_drop = df[(df['cweight'] == 0) & (df['dweight'] == 0)]
        df = df.drop(df_drop.index)

        # handing missing data
        df_complete = df.copy()
        df_complete = df_complete[
            (df_complete['cweight'] != 0) & (df_complete['dweight'] != 0)]
        aver_min = (df_complete['dweight'].sum(
        ) - df_complete['cweight'].sum()) / df_complete.shape[0]
        df['dweight'] = df.apply(
            lambda row: row['dweight'] if row[
                'dweight'] != 0 else round(row['cweight'] + aver_min, 1),
            axis=1)
        df['cweight'] = df.apply(
            lambda row: row['cweight'] if row[
                'cweight'] != 0 else round(row['dweight'] - aver_min, 1),
            axis=1)

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
        df.to_csv('transformed_dataset/input.csv')

        return df

    def reformat_output(self):
        df = pd.read_csv(self.output_path)
        df = df.fillna(value={'anti_add': 0})

        df = df.dropna(subset=['mode', 'machine', 'anti_type', 'anti_first'])

        # first try: don not include flow in output
        df = df.drop(columns=['flow'])
        df.to_csv('transformed_dataset/output.csv')
        return df

    def reformat(self):
        input_data = self.reformat_input()
        output_data = self.reformat_output()

        df_concat = pd.merge(input_data, output_data,
                             how='left', left_on='id', right_on='id')
        df_concat = df_concat.dropna()
        df_concat = df_concat.drop(columns=['id'])

        df_concat.to_csv('transformed_dataset/final.csv', index=False)


if __name__ == '__main__':
    preprocessing = Preprocessing('dataset/input.csv', 'dataset/output.csv')
    preprocessing.reformat()
