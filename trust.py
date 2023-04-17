# Bruno Iochins Grisci
# April 17th, 2023
#
import os
import sys
import importlib
import pandas as pd

from openTSNE import TSNE
import numpy as np
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.manifold import trustworthiness
import matplotlib.pyplot as plt
from matplotlib.colors import is_color_like
import plotly.tools as tls
import plotly.express as px
import utilstsne
from tqdm import tqdm

import RR_utils

config_file = sys.argv[1]
print(config_file)
cfg = importlib.import_module(config_file.replace('/','.').replace('.py',''))
selector_file = sys.argv[2]
projection_file = sys.argv[3]

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

def main():
    
    print('Loading ' + cfg.dataset_file)
    # https://stackoverflow.com/questions/52209290/how-do-i-make-a-progress-bar-for-loading-pandas-dataframe-from-a-large-xlsx-file
    df = pd.concat([chunk for chunk in tqdm(pd.read_csv(cfg.dataset_file, delimiter=cfg.dataset_sep, header=0, index_col=cfg.row_index, chunksize=cfg.load_chunksize), desc='Loading data from {} in chunks of {}'.format(cfg.dataset_file, cfg.load_chunksize))])    
    # https://stackoverflow.com/questions/30494569/how-to-force-pandas-read-csv-to-use-float32-for-all-float-columns
    df = df.astype({c: cfg.dtype_float for c in df.select_dtypes(include='float64').columns})
    df = df.astype({c: cfg.dtype_int for c in df.select_dtypes(include='int64').columns})   
    df = RR_utils.check_dataframe(df, cfg.class_label, cfg.task)

    COL = {}
    CLASS_LABELS = list(np.sort(df[cfg.class_label].astype(str).unique()))
    if cfg.task == 'classification':
        if len(cfg.class_colors) < len(CLASS_LABELS):
            raise Exception('There are only {} colors for {} classes.'.format(len(cfg.class_colors), len(CLASS_LABELS)))
            
    for c, l in zip(cfg.class_colors, CLASS_LABELS):
        COL[l] = colhex.get(c,c)
    print(COL)

    if cfg.task == 'classification':
        utilstsne.check_colors(COL, colhex)    
    
    print('THE DATA SET:')
    print(df)
    print(df.dtypes) 
    
    if cfg.task == 'classification':
        classes = list(np.sort(df[cfg.class_label].astype(str).unique()))
        print('THE CLASSES:')
        print(classes)

        if len(classes) > len(cfg.class_colors):
            raise Exception("The number of colors does not match the number of classes!") 
    elif cfg.task == 'regression':
        classes = []
        targets_info = (df[cfg.class_label].min(), df[cfg.class_label].max(), df[cfg.class_label].mean(), df[cfg.class_label].std(), df[cfg.class_label].median())
        print('TARGETS INFO:')
        print(targets_info)

    if not os.path.exists(cfg.output_folder):
        os.makedirs(cfg.output_folder)

    if cfg.standardized:
        df, meanVals, stdVals = RR_utils.standardize(df, cfg.class_label)
        del meanVals
        del stdVals

    y = df[cfg.class_label].astype(str)    
    x = df.drop(cfg.class_label, axis=1).to_numpy()
    
    print(cfg.class_label)
    print(y)
    
    print("Data set contains %d samples with %d features" % x.shape)

    if selector_file != 'no':
        print('All weights as in %s' % selector_file)
        rel = pd.read_csv(selector_file, delimiter=cfg.dataset_sep, header=0, index_col=0)
        print(rel)
        #rel = rel.abs().div(rel.abs().max(axis=1), axis=0)
        W = rel.to_numpy()
        if W.shape[1] == 1:
            W = np.tile(W,(1, x.shape[0])).transpose()
            wx = np.multiply(W,x)
        else:
            wx = np.multiply(W,x) # wx = W * x
        print("-----")
        print(df)
        print(rel)
    else:
        wx = x
        
    proj = pd.read_csv(projection_file, delimiter=cfg.dataset_sep, header=0, index_col=0)
    print(proj)
    p2d = proj.drop(cfg.class_label, axis=1).to_numpy()
    #p2d = proj.drop('y', axis=1).to_numpy()
    
    print(wx)
    print(p2d)
    
    print(wx.shape)
    print(p2d.shape)
    
    trusty = trustworthiness(wx, p2d, n_neighbors=7, metric='euclidean')
    print(trusty)

if __name__ == '__main__': 
    main()
    
    #python3 trust.py config.py selections/Lasso_xor.csv DMDK/results/xor/Lasso_xor_tsne2d.csv 