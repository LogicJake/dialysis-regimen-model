# -*- coding: utf-8 -*-
# @Author: LogicJake
# @Date:   2018-11-12 10:00:06
# @Last Modified time: 2018-11-12 10:14:57
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


def plot_bar(data, title, xlabel, ylabel='number', path=None):
    if path is None:
        path = 'analysis' + os.sep
    else:
        if path[-1] != os.sep:
            path += os.sep
    if not os.path.exists(path):
        os.makedirs(path)

    plt.cla()
    plt.figure(figsize=(12, 6))

    index = data.index.tolist()
    new_index = [str(x) for x in index]

    index = new_index
    value = data.tolist()

    plt.bar(index, value, width=0.3, align='center', alpha=0.8)
    plt.xticks(index, size='small', rotation=20)
    for a, b in zip(index, value):
        plt.text(a, b + 0.05, '%.0f' %
                 b, ha='center', va='bottom', fontsize=10)

    plt.xlabel(xlabel)

    plt.ylabel(ylabel)

    plt.title(title)

    file_name = title.replace(' ', '_')
    plt.tight_layout()

    plt.savefig(path + file_name + '.png')


df = pd.read_csv('transformed_dataset/input.csv')
uid_count = df['uid'].value_counts()
# plot_bar(uid_count, 'user amount distribution', 'user amount')
total = np.sum(uid_count)
print(total)
