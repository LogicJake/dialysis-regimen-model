# -*- coding: utf-8 -*-
# @Author: LogicJake
# @Date:   2018-11-11 15:37:32
# @Last Modified time: 2018-11-14 10:56:27
import os
import pandas as pd
import numpy as np

mm_top = 100
anti_top = 100
series_length = 7


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
        df = df[['id', 'uid', 'sex', 'age', 'disease',
                 'complication', 'dweight', 'cweight', 'date']]

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
        df.to_csv('transformed_dataset/input.csv', index=False)

        return df

    def reformat_output(self):
        df = pd.read_csv(self.output_path)
        df = df.fillna(value={'anti_add': 0})

        df = df.dropna(subset=['mode', 'machine', 'anti_type', 'anti_first'])

        # first try: don not include flow in output
        df = df.drop(columns=['flow'])

        # merge output
        df['mm'] = df['machine'].str.cat(df['mode'], sep=':')
        df = df.drop(['machine', 'mode'], axis=1)
        df['anti_first'] = df['anti_first'].map(lambda x: str(x))
        df['anti'] = df['anti_type'].str.cat(df['anti_first'], sep=':')
        df = df.drop(['anti_first', 'anti_type'], axis=1)

        mm_count = df['mm'].value_counts()
        mm_preserve = mm_count[:mm_top].index.tolist()
        anti_count = df['anti'].value_counts()
        anti_preserve = anti_count[:anti_top].index.tolist()

        df['mm'] = df['mm'].map(lambda x: x if x in mm_preserve else 'other')
        df['anti'] = df['anti'].map(
            lambda x: x if x in anti_preserve else 'other')
        df.to_csv('transformed_dataset/output.csv', index=False)
        return df

    def reformat(self):
        input_data = self.reformat_input()
        output_data = self.reformat_output()

        df_concat = pd.merge(input_data, output_data,
                             how='left', left_on='id', right_on='id')
        df_concat = df_concat.dropna()
        df_concat = df_concat.drop(columns=['id', 'anti_add'])

        df_series = self.time_series(df_concat)
        df_series.to_csv('transformed_dataset/final.csv', index=False)

    def time_series(self, data):
        mm_num = len(data['mm'].unique())
        anti_num = len(data['anti'].unique())

        encoded_columns = ['mm', 'anti']
        data = pd.get_dummies(data, columns=encoded_columns, prefix_sep='+')

        col_name = data.columns.values.tolist()
        input_col = col_name[1:]
        input_col.remove('date')
        output_col = col_name[-(mm_num + anti_num):]

        names = []
        for i in range(series_length, 0, -1):
            names += ['%s(t-%d)' % (col, i) for col in input_col]
        names += ['%s(t)' % col for col in output_col]

        data.sort_values('date', ascending=True, inplace=True)
        res = data.groupby('uid')

        entire_series = []
        for uid, group in res:
            length = group.shape[0]
            group = group.drop(['uid', 'date'], axis=1)
            if length >= series_length + 1:
                series = []
                for i in range(series_length + 1):
                    if i != series_length:
                        series.append(group.shift(-i))
                    else:
                        series.append(group[output_col].shift(-i))
                agg = pd.concat(series, axis=1)
                agg.dropna(inplace=True)
                entire_series.append(agg)

        reframed = pd.concat(entire_series)
        reframed.columns = names
        return reframed


if __name__ == '__main__':
    preprocessing = Preprocessing(
        '../dataset/input.csv', '../dataset/output.csv')
    preprocessing.reformat()
