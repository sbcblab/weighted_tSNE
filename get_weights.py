# Bruno Iochins Grisci
# February 22nd, 2021

import os
import sys
import importlib
import pandas as pd
import numpy as np
import ast

if __name__ == '__main__': 

    dataset_file = 'DATA/selection.csv'

    df = pd.read_csv(dataset_file, delimiter=',', header=0, index_col=None)
    df = df[df['dataset_name'] == 'Liver_GSE22405']
    df = df[df['sampling'] == 'none']

    print(df)
    print(df['name'].unique())

    df.drop_duplicates('name', inplace = True)


    df['values'] = df['values'].apply(ast.literal_eval)
    df['values'] = df['values'].apply(np.array)
    print(df)

    for i in range(11):
        a = df.iloc[i]
        print(a)
        file_name = 'DATA/' + a['dataset_name'] + a['name'] + '.csv'
        if a['result_type'] == 'weights':
            if a['name'] == 'KruskallWallisFilter':
                w = pd.DataFrame({'weights': 1.0 - a['values']})
            else:
                w = pd.DataFrame({'weights': a['values']})
        elif a['result_type'] == 'rank':
            w = pd.DataFrame({'weights': (a['values'].max() - a['values'])/a['values'].max()})
        elif a['result_type'] == 'subset':
            arr = np.zeros(22283)
            for one in a['values']:
                arr[one] = 1.0
            w = pd.DataFrame({'weights': arr})
        w.to_csv(file_name)

