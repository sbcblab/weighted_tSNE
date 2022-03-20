# Bruno Iochins Grisci
# March 17th, 2022

import os
import sys
import importlib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

import RR_utils

config_file = sys.argv[1]
cfg = importlib.import_module(config_file.replace('/','.').replace('.py',''))

def get_acc(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    #clf = GaussianNB()
    #clf = LogisticRegression(random_state=42)
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    errors = (y_test != y_pred).sum()
    acc = (y_test.shape[0] - errors) / y_test.shape[0]
    print("Number of mislabeled points out of a total %d points : %d" % (x_test.shape[0], (y_test != y_pred).sum()))   
    print(acc) 
    y_test = np.unique(y_test, return_inverse=True)[1]
    y_pred = np.unique(y_pred, return_inverse=True)[1]
    #met = roc_auc_score(y_test, y_pred, average='weighted')
    met = f1_score(y_test, y_pred, average='weighted')
    print(met)
    return met

def main():

    df = pd.read_csv(cfg.dataset_file, delimiter=cfg.dataset_sep, header=0, index_col=cfg.row_index)
    df = RR_utils.check_dataframe(df, cfg.class_label, cfg.task)

    print('THE DATA SET:')
    print(df)

    if cfg.task == 'classification':
        classes = list(np.sort(df[cfg.class_label].astype(str).unique()))
        print('THE CLASSES:')
        print(classes)

        if len(classes) > len(cfg.class_colors):
            raise Exception("The number of colors does not match the number of classes!") 
    elif cfg.task == 'regression':
        classes = []
        targets_info = (df[cfg.class_label].min(), df[cfg.class_label].max(), df[cfg.class_label].mean(), df[cfg.class_label].std())
        print('TARGETS INFO:')
        print(targets_info)

    if not os.path.exists(cfg.output_folder):
        os.makedirs(cfg.output_folder)

    if cfg.standardized:
        df, meanVals, stdVals = RR_utils.standardize(df, cfg.class_label)

    sa = {}
    x = df.drop(cfg.class_label, axis=1).to_numpy()
    y = df[cfg.class_label].astype(str)
    acc = get_acc(x, y)
    sa['no scoring'] = acc

    for selector_file in cfg.weights_file:
        print(selector_file)
        rel = pd.read_csv(selector_file, delimiter=cfg.dataset_sep, header=0, index_col=0)
        '''
        sel = int(rel.shape[0]*0.01)
        sel = 5
        print(rel.shape[0], sel)
        nrel = rel.nlargest(sel, 'value')
        nrel = nrel[nrel['value'] > 0.0]
        print(nrel)
        print(nrel.shape)
        x = df[nrel.index].to_numpy()
        y = df[cfg.class_label].astype(str)
        acc = get_acc(x, y)
        sa[selector_file] = acc
        '''
        W = rel.to_numpy()
        if W.shape[1] == 1:
            W = np.tile(W,(1, x.shape[0])).transpose()
            wx = np.multiply(W,x)
        else:
            wx = np.multiply(W,x) # wx = W * x
        acc = get_acc(wx, y)     
        sa[selector_file] = acc
    dfsa = pd.DataFrame.from_dict(sa, orient='index', columns=['F1-score'])
    print(dfsa)

if __name__ == '__main__': 
    main()