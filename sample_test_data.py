# -*- coding: utf-8 -*-
# @Author: LogicJake
# @Date:   2018-11-18 14:34:40
# @Last Modified time: 2018-11-18 15:39:23
# from dataset sample data for testing
import pandas as pd
import numpy as np
import os
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


def sample_ts():
    pass


if __name__ == '__main__':
    sample(100)
    sample_ts()
