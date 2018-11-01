# -*- coding: utf-8 -*-
# @Author: LogicJake
# @Date:   2018-11-01 10:01:08
# @Last Modified time: 2018-11-01 15:58:42
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


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


if __name__ == '__main__':
    df = pd.read_csv('transformed_dataset/output.csv')

    mode_count = df['mode'].value_counts()
    machine_count = df['machine'].value_counts()
    anti_type_count = df['anti_type'].value_counts()
    anti_first_count = df['anti_first'].value_counts()
    anti_add_count = df['anti_add'].value_counts()

    plot_bar(mode_count, 'mode distribution', 'mode')
    plot_bar(machine_count, 'machine distribution', 'machine')
    plot_bar(anti_type_count, 'anti_type distribution', 'anti_type')
    plot_bar(anti_first_count, 'anti_first distribution', 'anti_first')
    plot_bar(anti_add_count, 'anti_add distribution', 'anti_add')

    machines = df['machine'].drop_duplicates().tolist()
    for machine in machines:
        data = df.loc[df['machine'] == machine]['mode'].value_counts()
        plot_bar(data, 'mode distribution under ' +
                 machine, 'mode', path='analysis/mode_distribution_under_machine')

    modes = df['mode'].drop_duplicates().tolist()
    for mode in modes:
        data = df.loc[df['mode'] == mode]['machine'].value_counts()
        plot_bar(data, 'machine distribution under ' +
                 mode, 'machine', path='analysis/machine_distribution_under_mode')

    anti_types = df['anti_type'].drop_duplicates().tolist()
    for anti_type in anti_types:
        data = df.loc[df['anti_type'] == anti_type][
            'anti_first'].value_counts()
        plot_bar(data, 'anti_first distribution under ' +
                 anti_type, 'anti_first', path='analysis/anti_first_distribution_under_anti_type')

    anti_firsts = df['anti_first'].drop_duplicates().tolist()
    for anti_first in anti_firsts:
        data = df.loc[df['anti_first'] == anti_first][
            'anti_type'].value_counts()
        plot_bar(data, 'anti_type distribution under ' +
                 str(anti_first), 'anti_type', path='analysis/anti_type_distribution_under_anti_first')
