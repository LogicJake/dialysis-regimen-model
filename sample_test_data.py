# -*- coding: utf-8 -*-
# @Author: LogicJake
# @Date:   2018-11-18 14:34:40
# @Last Modified time: 2018-11-18 20:35:16
# from dataset sample data for testing
import pandas as pd
import numpy as np
import os
import random
from LSTM.preprocessing import series_length
save_path = 'test_data' + os.path.sep


def sample(n=10, dweight_frac=0.2):
    input_path = 'dataset' + os.path.sep + 'input.csv'
    output_path = 'dataset' + os.path.sep + 'output.csv'

    df_input = pd.read_csv(input_path)
    df_output = pd.read_csv(output_path)

    df_input = df_input[(df_input['sex'].str.len() > 0) &
                        (df_input['dweight'] != 0) & (df_input['cweight'] != 0)]
    df_input['disease'] = df_input['disease'].map(lambda x: np.nan if type(
        x) != float and (len(x) < 2 or x.isspace()) else x)
    df_input['complication'] = df_input['complication'].map(
        lambda x: np.nan if type(x) != float and (len(x) < 2 or x.isspace()) else x)

    df_input = df_input.drop(['uid', 'date'], axis=1)
    df_input = df_input.sample(n=n)

    select_index = df_input.index.values.tolist()
    df_output = df_output.loc[select_index]
    df_output = df_output.drop(['anti_add'], axis=1)

    col_name = df_output.columns.tolist()
    col_name.insert(1, 'dweight')
    df_output = df_output.reindex(columns=col_name)
    df_output['dweight'] = df_input['dweight']

    # sample data that not having dweight
    sample_index = df_input.sample(frac=dweight_frac)['id'].values.tolist()
    df_input['dweight'] = df_input.apply(
        lambda row: 0 if row['id'] in sample_index else row['dweight'], axis=1)

    df_output['dweight'] = df_output.apply(
        lambda row: row['dweight'] if row['id'] in sample_index else np.nan, axis=1)

    df_input = df_input.drop(['id'], axis=1)
    df_output = df_output.drop(['id'], axis=1)

    df_input.to_csv(save_path + 'example1.csv', index=False)
    df_output.to_csv(save_path + 'label1.csv', index=False)


def sample_lstm(n=10):
    input_path = 'dataset' + os.path.sep + 'input.csv'
    output_path = 'dataset' + os.path.sep + 'output.csv'

    df_input = pd.read_csv(input_path)
    df_output = pd.read_csv(output_path)

    df_input = df_input[['id', 'uid', 'sex', 'age', 'disease',
                         'complication', 'dweight', 'cweight', 'date']]

    # change the blank value of disease and complication to nan
    df_input['disease'] = df_input['disease'].map(lambda x: np.nan if type(
        x) != float and (len(x) < 2 or x.isspace()) else x)
    df_input['complication'] = df_input['complication'].map(
        lambda x: np.nan if type(x) != float and (len(x) < 2 or x.isspace()) else x)

    # delete that the value of cweight and dweight are all 0
    df_input = df_input[(df_input['sex'].str.len() > 0) &
                        (df_input['dweight'] != 0) & (df_input['cweight'] != 0)]

    df_input.sort_values('date', ascending=True, inplace=True)
    res = list(df_input.groupby('uid'))

    num = 0
    input_sample_list = []
    output_sample_list = []
    while num != n:
        choice = random.randint(1, len(res)) - 1
        df_choice = res[choice][1]
        if df_choice.shape[0] < series_length + 1:
            continue
        start = random.randint(0, df_choice.shape[0] - series_length - 1)
        num += 1
        input_sample = df_choice[start:start + series_length + 1]
        output_sample = df_output.loc[input_sample.index.values.tolist()]
        df_concat = pd.merge(input_sample, output_sample,
                             how='left', left_on='id', right_on='id')
        input_sample_list.append(df_concat[0:series_length])
        output_sample_list.append(df_concat.iloc[-1:])

    sample_input = pd.concat(input_sample_list)
    sample_input = sample_input.drop(
        ['flow', 'date', 'anti_add', 'id'], axis=1)

    sample_output = pd.concat(output_sample_list)
    sample_output = sample_output[
        ['mode', 'machine', 'flow', 'anti_type', 'anti_first']]

    sample_input.to_csv(save_path + 'example2.csv', index=False)
    sample_output.to_csv(save_path + 'label2.csv', index=False)


if __name__ == '__main__':
    sample(100)
    sample_lstm(100)
