# -*- coding: utf-8 -*-
# @Author: LogicJake
# @Date:   2018-11-16 17:03:42
# @Last Modified time: 2018-11-18 20:57:09
import argparse
import os
import logging
import pandas as pd
import numpy as np
import model.DNN.model as dnn_m
import model.LSTM.model as lstm_m
import model.predict_dweight.model as pw_m
import model.predict_flow.model as pf_m
from model.LSTM.preprocessing import series_length

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s-%(name)s-%(levelname)s: %(message)s',
                    datefmt='%Y/%d/%m %H:%M:%S')


def parse_args():
    # Parses arguments

    parser = argparse.ArgumentParser(
        description="model for renal dialysis regimen")

    parser.add_argument('-p', '--path', type=str, required=True,
                        help='location of predicting file')

    parser.add_argument('-l', '--lstm', type=bool, default=False,
                        help='whether use LSTM model')
    return parser.parse_args()


def read_mapping():
    disease = None
    complication = None

    with open('model/feature_mapping.txt', 'r') as f:
        content = f.readline()
        content = eval(content)
        disease = content['disease']
        complication = content['complication']

    return disease, complication


def read_pf_mapping():
    machine = None
    mode = None
    anti_type = None
    anti_first = None

    with open('model/predict_flow/model/mapping.txt', 'r') as f:
        content = f.readline()
        content = eval(content)
        machine = content['machine']
        mode = content['mode']
        anti_type = content['anti_type']
        anti_first = content['anti_first']

    return machine, mode, anti_type, anti_first


def read_LSTM_mapping():
    mm = None
    anti = None

    with open('model/LSTM/model/labels.txt', 'r') as f:
        content = f.readline()
        content = eval(content)
        mm = content['mm']
        anti = content['anti']

    return mm, anti


def one_hot(df, name, names):
    for index, d in enumerate(names):
        df[name + '+' + str(index)] = df.apply(
            lambda row: 1 if d == row[name] else 0, axis=1)
    df = df.drop([name], axis=1)
    return df


def predict_flow(final_output):
    # predict flow
    machine, mode, anti_type, anti_first = read_pf_mapping()

    output_features = final_output.copy()
    output_features = output_features[
        ['machine', 'mode', 'anti_type', 'anti_first']]

    output_features = output_features[(output_features['machine'] != 'other') & (output_features[
        'mode'] != 'other') & (output_features['anti_type'] != 'other') & (output_features['anti_first'] != 'other')]
    if output_features.shape[0] != 0:
        output_features = one_hot(output_features,  'machine', machine)
        output_features = one_hot(output_features, 'mode', mode)
        output_features = one_hot(output_features, 'anti_type', anti_type)
        output_features = one_hot(output_features, 'anti_first', anti_first)

        pf_model = pf_m.Model()
        res = pf_model.predict(output_features)
        output_features['flow'] = res
        final_output['flow'] = output_features['flow']
    final_output = final_output.round({'dweight': 1})
    final_output['flow'].fillna('other', inplace=True)
    return final_output


def predict_dweight(df, final_output):
    dweight = df[(df['dweight'] == 0)]
    dweight = dweight.drop(['dweight'], axis=1)
    pw_model = pw_m.model()
    if dweight.shape[0] != 0:
        res = pw_model.predict(dweight)
        final_output['dweight'] = dweight['cweight'] + res
        df['dweight'] = final_output['dweight'].fillna(0) + df['dweight']


def predict(path):
    final_output = pd.DataFrame(
        columns=['dweight', 'mode', 'machine', 'flow', 'anti_type', 'anti_first'])

    df = pd.read_csv(path)
    final_output['dweight'] = df['dweight'].copy()
    final_output['dweight'] = np.nan

    df = df[['sex', 'age', 'disease', 'complication', 'dweight', 'cweight']]
    disease, complication = read_mapping()
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

    # one-hot
    for index, d in enumerate(disease):
        df['d_' + str(index)] = df.apply(
            lambda row: 1 if type(row['disease']) == str and d in row['disease'] else 0, axis=1)
    df = df.drop(['disease'], axis=1)

    for index, c in enumerate(complication):
        df['c_' + str(index)] = df.apply(
            lambda row: 1 if type(row['complication']) == str and c in row['complication'] else 0, axis=1)
    df = df.drop(['complication'], axis=1)

    predict_dweight(df, final_output)

    # predict labels except flow
    DNN_model = dnn_m.DNNModel()
    mm, anti = DNN_model.predict(df.values)
    machine = []
    mode = []
    anti_type = []
    anti_first = []
    for a in mm:
        if a == 'other':
            machine.append(a)
            mode.append(a)
        else:
            machine.append(a.split('*')[0])
            mode.append(a.split('*')[1])
    for a in anti:
        if a == 'other':
            anti_type.append(a)
            anti_first.append(a)
        else:
            anti_type.append(a.split('*')[0])
            anti_first.append(a.split('*')[1])

    final_output['mode'] = mode
    final_output['machine'] = machine
    final_output['anti_type'] = anti_type
    final_output['anti_first'] = anti_first
    final_output = predict_flow(final_output)
    final_output.to_csv('test_data' + os.path.sep + 'result1.csv', index=False)


def predict_lstm(path):
    final_output = pd.DataFrame(
        columns=['mode', 'machine', 'flow', 'anti_type', 'anti_first'])

    df = pd.read_csv(path)
    assert(df.shape[0] % series_length == 0)
    n = df.shape[0] / series_length
    n = int(n)

    df_1 = df[['sex',  'age', 'disease', 'complication', 'dweight', 'cweight']]
    df_2 = df[['mode', 'machine', 'anti_type', 'anti_first']]

    sex_mapping = {
        '女': 0,
        '男': 1
    }

    # change the value of sex to number
    df_1['sex'] = df_1['sex'].map(sex_mapping)

    disease, complication = read_mapping()
    for index, d in enumerate(disease):
        df_1['d_' + str(index)] = df_1.apply(
            lambda row: 1 if type(row['disease']) == str and d in row['disease'] else 0, axis=1)
    df_1 = df_1.drop(['disease'], axis=1)

    for index, c in enumerate(complication):
        df_1['c_' + str(index)] = df_1.apply(
            lambda row: 1 if type(row['complication']) == str and c in row['complication'] else 0, axis=1)
    df_1 = df_1.drop(['complication'], axis=1)

    df_2['mm'] = df_2['machine'].str.cat(df_2['mode'], sep='*')
    df_2 = df_2.drop(['machine', 'mode'], axis=1)
    df_2['anti_first'] = df_2['anti_first'].map(lambda x: str(x))
    df_2['anti'] = df_2['anti_type'].str.cat(df_2['anti_first'], sep='*')
    df_2 = df_2.drop(['anti_first', 'anti_type'], axis=1)

    mm, anti = read_LSTM_mapping()
    for index, c in mm.items():
        df_2['mm?' + str(c)] = df_2.apply(
            lambda row: 1 if type(row['mm']) == str and c in row['mm'] else 0, axis=1)
    for index, c in anti.items():
        df_2['anti?' + str(c)] = df_2.apply(
            lambda row: 1 if type(row['anti']) == str and c in row['anti'] else 0, axis=1)
    df_2 = df_2.drop(['mm'], axis=1)
    df_2 = df_2.drop(['anti'], axis=1)
    df = pd.concat([df_1, df_2], axis=1)

    aggs = []
    for i in range(n):
        start = i * series_length
        end = (i + 1) * series_length
        g = df.iloc[start:end]

        series = []
        for i in range(series_length):
            series.append(g.shift(-i))
        agg = pd.concat(series, axis=1)
        agg.dropna(inplace=True)
        aggs.append(agg)

    df_aggs = pd.concat(aggs)
    lstm_model = lstm_m.LSTMModel()

    X = df_aggs.values
    X = X.reshape(
        (X.shape[0], series_length, int(X.shape[1] / series_length)))
    mm, anti = lstm_model.predict(X)
    machine = []
    mode = []
    anti_type = []
    anti_first = []
    for a in mm:
        machine.append(a.split('*')[0])
        mode.append(a.split('*')[1])
    for a in anti:
        anti_type.append(a.split('*')[0])
        anti_first.append(a.split('*')[1])
    final_output['mode'] = mode
    final_output['machine'] = machine
    final_output['anti_type'] = anti_type
    final_output['anti_first'] = anti_first
    predict_flow(final_output)
    final_output = predict_flow(final_output)
    final_output.to_csv('test_data' + os.path.sep + 'result2.csv', index=False)

if __name__ == "__main__":
    args = parse_args()

    lstm = args.lstm
    path = args.path
    if not os.path.exists(path):
        logging.error("'%s' does not exist", path)
        exit(-1)
    if not lstm:
        predict(path)
    else:
        predict_lstm(path)
