# Bruno Iochins Grisci
# February 18st, 2021

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

dummy = False
#dummy = 'zero'
#dummy = 'one'
#dummy = 50

if __name__ == '__main__': 

    df = pd.read_csv(cfg.dataset_file, delimiter=cfg.dataset_sep, header=0, index_col=cfg.row_index)
    df = RR_utils.check_dataframe(df, cfg.class_label, cfg.task)
    
    print('THE DATA SET:')

    print(df)

    if not os.path.exists(cfg.output_folder):
        os.makedirs(cfg.output_folder)

    if cfg.standardized:
        df, meanVals, stdVals = RR_utils.standardize(df, cfg.class_label)

    x = df.drop(cfg.class_label, axis=1).to_numpy()
    y = df[cfg.class_label].astype(str)

    print("Data set contains %d samples with %d features" % x.shape)

    original_silhouette = metrics.silhouette_score(x, y, metric='euclidean')

    if not cfg.no_weights:
        if dummy == False:
            print('All weights as in %s' % cfg.weights_file)
            rel = pd.read_csv(cfg.weights_file, delimiter=cfg.dataset_sep, header=0, index_col=0)
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
        elif dummy == 'zero':
            print('All weights = 0')
            weights = np.zeros(x.shape[1])
            wx = weights * x
            print(weights)
            print(weights.shape)
        elif dummy == 'one':
            print('All weights = 1')
            weights = np.ones(x.shape[1])
            wx = weights * x
            print(weights)
            print(weights.shape)    
        else:
            print('%d weights = 1 else 0' % dummy)
            weights = np.zeros(x.shape[1])
            for d in range(dummy):
                weights[d] = 1
            W = np.tile(weights, (x.shape[0],1))
            wx = weights * x
            print(weights)
            print(weights.shape)
            print(W)
            print(W.shape)

        print(wx)
        print(wx.shape)
    else:
        wx = x

    perp = 0
    if cfg.perplexity == "auto":
        perp = max(30, x.shape[0]/100)
    else:
        perp = cfg.perplexity

    tsne = TSNE(
        perplexity=perp,
        initialization="pca",
        metric='euclidean',
        neighbors="auto",
        learning_rate="auto",
        verbose=True,
        n_jobs=8,
        n_iter=cfg.n_iter,
        random_state=42,
    )

    embedding = tsne.fit(wx)

    print(embedding)

    emb = pd.DataFrame(data=embedding,    # values
                       index=df.index,    # 1st column as index
                       columns=['wt-SNE 1', 'wt-SNE 2'])  # 1st row as the column names
    emb[cfg.class_label] = y
    emb.to_csv(os.path.basename(cfg.dataset_file).replace('.csv','_tsne.csv'))

    print(emb)

    tsne_silhouette = metrics.silhouette_score(embedding, y, metric='euclidean')

    COL = {}
    CLASS_LABELS = list(np.sort(df[cfg.class_label].astype(str).unique()))
    for c, l in zip(cfg.class_colors, CLASS_LABELS):
        COL[l] = colhex[c]
    print(COL)

    ax = utilstsne.plot(embedding, y, task=cfg.task, class_label=cfg.class_label, colors=COL, title=os.path.basename(cfg.weights_file).replace('.csv',''), draw_centers=True, draw_cluster_labels=False, s=cfg.dot_size)
    plt.savefig(cfg.output_folder + os.path.basename(cfg.weights_file).replace('.csv','_tsne.pdf'), bbox_inches='tight')

    print('Original silhouette', original_silhouette)
    print('t-SNE silhouette', tsne_silhouette)

