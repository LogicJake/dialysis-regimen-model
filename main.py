# -*- coding: utf-8 -*-
# @Author: LogicJake
# @Date:   2018-11-16 17:03:42
# @Last Modified time: 2018-11-18 14:07:06
import argparse
import os
import logging
import pandas as pd
import numpy as np
# import DNN.preprocessing as dnn_p
import DNN.model as dnn_m
# import LSTM.preprocessing as lstm_p
# import LSTM.model as lstm_m

import predict_dweight.model as pw_m
# import predict_flow.preprocessing as pf_p
import predict_flow.model as pf_m

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s-%(name)s-%(levelname)s: %(message)s',
                    datefmt='%Y/%d/%m %H:%M:%S')


def parse_args():
    # Parses arguments

    parser = argparse.ArgumentParser(
        description="model for renal dialysis regimen")
    parser.add_argument('-m', '--mode', type=str,
                        choices=['file', 'array'], required=True)
    parser.add_argument('-p', '--path', type=str,
                        help='location of predicting file')
    return parser.parse_args()


def read_mapping():
    disease = None
    complication = None

    with open('feature_mapping.txt', 'r') as f:
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

    with open('predict_flow/model/mapping.txt', 'r') as f:
        content = f.readline()
        content = eval(content)
        machine = content['machine']
        mode = content['mode']
        anti_type = content['anti_type']
        anti_first = content['anti_first']

    return machine, mode, anti_type, anti_first


def one_hot(df, name, names):
    for index, d in enumerate(names):
        df[name + '+' + str(index)] = df.apply(
            lambda row: 1 if d == row[name] else 0, axis=1)
    df = df.drop([name], axis=1)
    return df


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

    # predict dweight
    dweight = df[(df['dweight'] == 0)]
    dweight = dweight.drop(['dweight'], axis=1)
    pw_model = pw_m.model()
    res = pw_model.predict(dweight)
    final_output['dweight'] = dweight['cweight'] + res
    df['dweight'] = final_output['dweight'].fillna(0) + df['dweight']

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

    # predict flow
    machine, mode, anti_type, anti_first = read_pf_mapping()

    output_features = final_output.copy()
    output_features = output_features[
        ['machine', 'mode', 'anti_type', 'anti_first']]
    # just for test
    # output_features['machine'][2:8] = '德朗透析器B-14P'
    # output_features['mode'][2:8] = 'HD'
    # output_features['anti_type'][2:8] = '活多史4250u(剂量单位:u)'
    # output_features['anti_first'][2:8] = '4250.0'

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
    final_output.to_csv('predict_result.csv', index=False)


if __name__ == "__main__":
    args = parse_args()
    mode = args.mode

    if mode == 'file':
        path = args.path
        if not os.path.exists(path):
            logging.error("'%s' does not exist", path)
            exit(-1)
        predict(path)
    elif mode == 'array':
        pass
