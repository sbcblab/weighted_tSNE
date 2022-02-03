# Bruno Iochins Grisci
# February 18st, 2021
#
import os
import sys
import importlib
import pandas as pd

from openTSNE import TSNE
from openTSNE.callbacks import ErrorLogger
import numpy as np
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import plotly.tools as tls
import plotly.express as px
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

def get_pca(x, df, y, file_to_save, class_label, classes, colors, original_silhouette, weighted_silhouette):
    pca = PCA(n_components=cfg.n_components)
    components = pca.fit_transform(x)
    var = pca.explained_variance_ratio_

    pca_columns_labels = ['PC {} ({})'.format(i+1, var[i]) for i in range(cfg.n_components)]

    pca_emb = pd.DataFrame(data=components,    # values
                       index=df.index,    # 1st column as index
                       columns=pca_columns_labels)  # 1st row as the column names
    pca_emb[cfg.class_label] = y
    pca_emb.to_csv(cfg.output_folder + os.path.basename(file_to_save).replace('.csv','_pca{}d.csv'.format(cfg.n_components)))
    print(pca_emb)

    zl = None
    if cfg.n_components > 2:
        zl = pca_columns_labels[2]    
    fig = utilstsne.iteractive_plot(pca_emb, x_label=pca_columns_labels[0], y_label=pca_columns_labels[1], z_label=zl, sorted_classes={class_label: classes}, color_label=cfg.class_label, colors=[colhex[c] for c in cfg.class_colors], size=cfg.dot_size)
    fig.write_html(cfg.output_folder + os.path.basename(file_to_save).replace('.csv','_pca{}d.html'.format(cfg.n_components)))
    if cfg.show_figs:
        fig.show()

    ax = utilstsne.plot(components, y, task=cfg.task, class_label=cfg.class_label, colors=colors, title=os.path.basename(file_to_save).replace('.csv',''), draw_centers=True, draw_cluster_labels=False, s=cfg.dot_size)
    plt.savefig(cfg.output_folder + os.path.basename(file_to_save).replace('.csv','_pca{}d.pdf'.format(cfg.n_components)), bbox_inches='tight')

    embedding_silhouette = metrics.silhouette_score(components, y, metric='euclidean')
    silhouettes = {'Original silhouette': [original_silhouette], 'Weighted silhouette': [weighted_silhouette], 'Embedding silhouette': [embedding_silhouette]}
    s_df = pd.DataFrame.from_dict(silhouettes)
    print(s_df)
    s_df.to_csv(cfg.output_folder + os.path.basename(file_to_save).replace('.csv','silhouette_pca{}d.csv'.format(cfg.n_components)))

def main():
    df = pd.read_csv(cfg.dataset_file, delimiter=cfg.dataset_sep, header=0, index_col=cfg.row_index)
    df = RR_utils.check_dataframe(df, cfg.class_label, cfg.task)
    
    if cfg.no_weights:
        file_to_save = cfg.dataset_file
    else:
        file_to_save = cfg.weights_file

    COL = {}
    CLASS_LABELS = list(np.sort(df[cfg.class_label].astype(str).unique()))
    for c, l in zip(cfg.class_colors, CLASS_LABELS):
        COL[l] = colhex[c]
    print(COL)

    print('THE DATA SET:')
    print(df)

    classes = list(np.sort(df[cfg.class_label].astype(str).unique()))
    print('THE CLASSES:')
    print(classes)

    if len(classes) > len(cfg.class_colors):
        raise Exception("The number of colors does not match the number of classes!") 

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

    weighted_silhouette = metrics.silhouette_score(wx, y, metric='euclidean')

    if cfg.compute_pca:
        get_pca(wx, df, y, file_to_save, cfg.class_label, classes, COL, original_silhouette, weighted_silhouette)

    perp = 0
    if cfg.perplexity == "auto":
        perp = max(30, x.shape[0]/100)
    else:
        perp = cfg.perplexity

    ngm = "fft"
    if cfg.n_components > 2:
        ngm = 'barnes-hut'
    tsne = TSNE(
        n_components=cfg.n_components,
        perplexity=perp,
        initialization="pca",
        metric='euclidean',
        neighbors="auto",
        learning_rate="auto",
        negative_gradient_method=ngm,
        verbose=True,
        n_jobs=8,
        n_iter=cfg.n_iter,
        random_state=42,
    )

    embedding = tsne.fit(wx)
    print(embedding)
    if cfg.rotation and embedding.shape[1] == 2:
        embedding = utilstsne.rotate(embedding)
        print(embedding)

    emb_columns_labels = ['wt-SNE {}'.format(i+1) for i in range(cfg.n_components)]
    emb = pd.DataFrame(data=embedding,    # values
                       index=df.index,    # 1st column as index
                       columns=emb_columns_labels)  # 1st row as the column names
    emb[cfg.class_label] = y
    emb.to_csv(cfg.output_folder + os.path.basename(file_to_save).replace('.csv','_tsne{}d.csv'.format(cfg.n_components)))
    print(emb)

    embedding_silhouette = metrics.silhouette_score(embedding, y, metric='euclidean')

    ax = utilstsne.plot(embedding, y, task=cfg.task, class_label=cfg.class_label, colors=COL, title=os.path.basename(file_to_save).replace('.csv',''), draw_centers=True, draw_cluster_labels=False, s=cfg.dot_size)
    plt.savefig(cfg.output_folder + os.path.basename(file_to_save).replace('.csv','_tsne{}d.pdf'.format(cfg.n_components)), bbox_inches='tight')

    if cfg.task == 'classification' and cfg.n_components > 1:
        zl = None
        if cfg.n_components > 2:
            zl = emb_columns_labels[2]
        plotly_fig = utilstsne.iteractive_plot(emb, x_label=emb_columns_labels[0], y_label=emb_columns_labels[1], z_label=zl, sorted_classes={cfg.class_label: classes}, color_label=cfg.class_label, colors=[colhex[c] for c in cfg.class_colors], size=cfg.dot_size)
        plotly_fig.write_html(cfg.output_folder + os.path.basename(file_to_save).replace('.csv','_tsne{}d.html'.format(cfg.n_components)))
        if cfg.show_figs:
            plotly_fig.show()

    silhouettes = {'Original silhouette': [original_silhouette], 'Weighted silhouette': [weighted_silhouette], 'Embedding silhouette': [embedding_silhouette]}
    s_df = pd.DataFrame.from_dict(silhouettes)
    print(s_df)
    s_df.to_csv(cfg.output_folder + os.path.basename(file_to_save).replace('.csv','silhouette_tsne{}d.csv'.format(cfg.n_components)))


    selectors = ['DecisionTree', 'KruskallWallisFilter', 'Lasso', 'LinearSVM', 
                 'MRMR', 'MutualInformationFilter', 'RandomForest', 'ReliefFFeatureSelector',
                 'ReliefFGeneticAlgorithm', 'SVMGeneticAlgorithm', 'SVMRFE']    

    plt.close()

if __name__ == '__main__': 
    main()

