# -*- coding: utf-8 -*-
# @Time    : 18-10-27 上午10:31
# @Author  : LogicJake
# @File    : preprocessing.py
import os
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

mm_top = 5
anti_top = 6

pwd = os.path.abspath(os.path.dirname(__file__))
td_path = pwd + os.path.sep + 'transformed_dataset' + os.path.sep


def create_dir():
    tmp_dir = pwd + os.path.sep + 'transformed_dataset'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)


class Preprocessing(object):

    def __init__(self, input_path, output_path):
        super(Preprocessing, self).__init__()

        self.input_path = input_path
        self.output_path = output_path
        create_dir()

    def read_mapping(self):
        disease = None
        complication = None

        with open('../feature_mapping.txt', 'r') as f:
            content = f.readline()
            content = eval(content)
            disease = content['disease']
            complication = content['complication']

        return disease, complication

    def reformat_input(self):
        disease, complication = self.read_mapping()

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
        for index, d in enumerate(disease):
            df['d_' + str(index)] = df.apply(
                lambda row: 1 if type(row['disease']) == str and d in row['disease'] else 0, axis=1)
        df = df.drop(['disease'], axis=1)

        for index, c in enumerate(complication):
            df['c_' + str(index)] = df.apply(
                lambda row: 1 if type(row['complication']) == str and c in row['complication'] else 0, axis=1)

        df = df.drop(['complication'], axis=1)
        df.to_csv(td_path + 'input.csv', index=False)

        return df

    def reformat_output(self):
        df = pd.read_csv(self.output_path)
        df = df.fillna(value={'anti_add': 0})

        df = df.dropna(subset=['mode', 'machine', 'anti_type', 'anti_first'])

        # first try: don not include flow in output
        df = df.drop(columns=['flow'])

        # merge output
        df['mm'] = df['machine'].str.cat(df['mode'], sep='*')
        df = df.drop(['machine', 'mode'], axis=1)
        df['anti_first'] = df['anti_first'].map(lambda x: str(x))
        df['anti'] = df['anti_type'].str.cat(df['anti_first'], sep='*')
        df = df.drop(['anti_first', 'anti_type'], axis=1)

        mm_count = df['mm'].value_counts()
        mm_preserve = mm_count[:mm_top].index.tolist()
        anti_count = df['anti'].value_counts()
        anti_preserve = anti_count[:anti_top].index.tolist()

        df['mm'] = df['mm'].map(lambda x: x if x in mm_preserve else 'other')
        df['anti'] = df['anti'].map(
            lambda x: x if x in anti_preserve else 'other')
        df.to_csv(td_path + 'output.csv', index=False)
        return df

    def reformat(self):
        input_data = self.reformat_input()
        output_data = self.reformat_output()

        df_concat = pd.merge(input_data, output_data,
                             how='left', left_on='id', right_on='id')
        df_concat = df_concat.dropna()
        df_concat = df_concat.drop(columns=['id'])

        # not sample data to test model
        df_test = df_concat.sample(frac=0.05, replace=False, axis=0)
        df_test = df_test.drop(columns=['anti_add'])
        df_test.to_csv(td_path + 'test.csv',
                       index=False, header=False)

        df_other = df_concat.drop(df_test.index)
        df_sample = self.oversample(df_other)
        # drop anti-add
        df_sample = df_sample.drop(columns=['anti_add'])
        df_sample.to_csv(td_path + 'final.csv',
                         index=False, header=False)

    def label2number(self, df, name):
        value = set()
        df.apply(lambda row: value.add(row[name]), axis=1)
        value = list(value)
        value.sort()
        trans = dict()
        reverse = dict()
        for index, m in enumerate(value):
            reverse[index] = m
            trans[m] = index
        df[name] = df[name].map(trans)
        return df, reverse

    def oversample(self, data):
        data, mm_reverse = self.label2number(data, 'mm')
        data, anti_reverse = self.label2number(data, 'anti')
        data = data.values

        # oversample mm
        X_index = list(range(39))
        X_index.append(40)
        X = data[:, X_index]
        Y = data[:, 39]

        sm = SMOTE(random_state=42)
        X_1, Y_1 = sm.fit_resample(X, Y)
        Y_1 = Y_1.reshape((Y_1.shape[0], -1))
        data = np.concatenate((X_1, Y_1), axis=1)

        # oversample anti
        X = data[:, X_index]
        Y = np.rint(data[:, 39])

        sm = SMOTE(random_state=42)
        X_1, Y_1 = sm.fit_resample(X, Y)
        Y_1 = Y_1.reshape((Y_1.shape[0], -1))
        data = np.concatenate((X_1, Y_1), axis=1)

        data = pd.DataFrame(data)
        data.columns = ['sex', 'age', 'dweight', 'cweight', 'd_0', 'd_1', 'd_2', 'd_3', 'd_4', 'd_5', 'd_6', 'd_7', 'd_8', 'd_9', 'd_10', 'd_11', 'd_12',
                        'd_13', 'd_14', 'd_15', 'd_16', 'c_0', 'c_1', 'c_2', 'c_3', 'c_4', 'c_5', 'c_6', 'c_7', 'c_8', 'c_9', 'c_10', 'c_11', 'c_12', 'c_13', 'c_14', 'c_15', 'c_16', 'anti_add', 'mm', 'anti']

        data['mm'] = data['mm'].map(lambda x: round(x))
        data['mm'] = data['mm'].map(mm_reverse)
        data['anti'] = data['anti'].map(anti_reverse)
        return data


if __name__ == '__main__':
    preprocessing = Preprocessing(
        '../dataset/input.csv', '../dataset/output.csv')
    preprocessing.reformat()
