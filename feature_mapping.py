# -*- coding: utf-8 -*-
# @Author: LogicJake
# @Date:   2018-11-17 15:03:56
# @Last Modified time: 2018-11-17 15:19:00
'''
this file products feature_mapping.txt
'''

import pandas as pd
import os
path = 'dataset' + os.path.sep + 'input.csv'


def feature_mapping():
    df = pd.read_csv(path)
    disease = []
    df.apply((lambda row: disease.extend(row['disease'].split(',')) if type(
        row['disease']) != float and len(row['disease']) > 1 else 1), axis=1)
    disease = list(set(disease))
    disease.sort()

    complication = []
    df.apply((lambda row: complication.extend(row['complication'].split(',')) if type(
        row['complication']) != float and len(row['complication']) > 1 else 1), axis=1)
    complication = list(set(complication))
    complication.sort()

    res = {}
    res['disease'] = disease
    res['complication'] = complication

    with open('feature_mapping.txt', 'w') as outfile:
        outfile.write(str(res))


if __name__ == '__main__':
    feature_mapping()
