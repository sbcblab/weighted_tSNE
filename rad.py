# Bruno Iochins Grisci
# July 30st, 2021

import os
import sys
import importlib
import pandas as pd

from openTSNE import TSNE
from openTSNE.callbacks import ErrorLogger
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import utilstsne

from yellowbrick.features import RadViz

import RR_utils

config_file = sys.argv[1]
cfg = importlib.import_module(config_file.replace('/','.').replace('.py',''))

colhex = {
    'RED':     '#BA0000',
    'BLUE':    '#0000FF',
    'YELLOW':  '#FFEE00',
    'GREEN':   '#048200',    
    'ORANGE':  '#FF6103',
    'BLACK':   '#000000',
    'CYAN':    '#00FFD4',    
    'SILVER':  '#c0c0c0',
    'MAGENTA': '#680082',
    'CREAM':   '#FFFDD0',
    'DRKBRW':  '#654321',
    'BEIGE':   '#C2C237',
    'WHITE':   '#FFFFFF',
}

if __name__ == '__main__': 

    df = pd.read_csv(cfg.dataset_file, delimiter=cfg.dataset_sep, header=0, index_col=cfg.row_index)
    df = RR_utils.check_dataframe(df, cfg.class_label, cfg.task)
    
    print('THE DATA SET:')
    print(df)

    print('NaN under an entire DataFrame:', df.isnull().sum().sum())

    classes = list(np.sort(df[cfg.class_label].astype(str).unique()))
    print('THE CLASSES:')
    print(classes)

    if len(classes) > len(cfg.class_colors):
        raise Exception("The number of colors does not match the number of classes!") 

    if not os.path.exists(cfg.output_folder):
        os.makedirs(cfg.output_folder)

    if cfg.standardized:
        df, meanVals, stdVals = RR_utils.standardize(df, cfg.class_label)

    df[cfg.class_label] = pd.Categorical(df[cfg.class_label], ordered=True).codes
    print(df)

    x = df.drop(cfg.class_label, axis=1)
    x = x.loc[:, (x != 0).any(axis=0)] # delete columns with only zero
    x = x.sample(frac=0.1, axis=1)
    y = df[cfg.class_label]

    print(x)
    print(y)
    print("Data set contains %d samples with %d features" % x.shape)

    original_silhouette = metrics.silhouette_score(x, y, metric='euclidean')
    print('Original silhouette', original_silhouette)

    # Instantiate the visualizer
    visualizer = RadViz(classes=classes, colors=[colhex[c] for c in cfg.class_colors])

    visualizer.fit(x, y)           # Fit the data to the visualizer
    visualizer.transform(x)        # Transform the data
    visualizer.show()              # Finalize and render the figure