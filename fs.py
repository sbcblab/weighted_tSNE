# Bruno Iochins Grisci
# April 4th, 2021

import os
import sys
import importlib
import pandas as pd

import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel

import RR_utils

config_file = sys.argv[1]
cfg = importlib.import_module(config_file.replace('/','.').replace('.py',''))

if __name__ == '__main__': 

    df = pd.read_csv(cfg.dataset_file, delimiter=cfg.dataset_sep, header=0, index_col=cfg.row_index)
    df = RR_utils.check_dataframe(df, cfg.class_label, cfg.task)

    print('THE DATA SET:')
    print(df)

    classes = list(np.sort(df[cfg.class_label].astype(str).unique()))
    print('THE CLASSES:')
    print(classes)

    if not os.path.exists(cfg.output_folder):
        os.makedirs(cfg.output_folder)

    if cfg.standardized:
        df, meanVals, stdVals = RR_utils.standardize(df, cfg.class_label)

    x = df.drop(cfg.class_label, axis=1).to_numpy()
    y = df[cfg.class_label].astype(str)

    print("Data set contains %d samples with %d features" % x.shape)

    clf = ExtraTreesClassifier(n_estimators=50)
    clf = clf.fit(x, y)
    scores = clf.feature_importances_

    w = pd.DataFrame({'value': scores}, index=df.drop(cfg.class_label, axis=1).columns)
    print(w)
    file_name = cfg.dataset_file.replace('.csv','') + 'tree' + '.csv'
    w.to_csv(file_name)
