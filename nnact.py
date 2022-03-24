# Bruno Iochins Grisci
# March 24th, 2022
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
import matplotlib.pyplot as plt
import plotly.tools as tls
import plotly.express as px
import utilstsne

import RR_utils


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

    dot_size       = 100    # size of dots in the plots
    class_colors   = ['RED', 'BLUE', 'YELLOW', 'CYAN', 'GREEN', 'BEIGE', 'DRKBRW', 'BLACK', 'SILVER', 'ORANGE']
    class_label = 'class'
    task = 'classification'
    dataset_file = 'IEEEVIS/liveract.csv'
    output_folder  = 'IEEEVIS/RESULTS/liver/'
    standardized = False

    df = pd.read_csv(dataset_file, delimiter=',', header=0, index_col=0)
    df = RR_utils.check_dataframe(df, class_label, task)

    COL = {}
    CLASS_LABELS = list(np.sort(df[class_label].astype(str).unique()))
    for c, l in zip(class_colors, CLASS_LABELS):
        COL[l] = colhex[c]
    print(COL)

    print('THE DATA SET:')
    print(df)

    if task == 'classification':
        classes = list(np.sort(df[class_label].astype(str).unique()))
        print('THE CLASSES:')
        print(classes)

        if len(classes) > len(class_colors):
            raise Exception("The number of colors does not match the number of classes!") 
    elif task == 'regression':
        classes = []
        targets_info = (df[class_label].min(), df[class_label].max(), df[class_label].mean(), df[class_label].std())
        print('TARGETS INFO:')
        print(targets_info)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if standardized:
        df, meanVals, stdVals = RR_utils.standardize(df, class_label)

    x = df.drop(class_label, axis=1).to_numpy()
    y = df[class_label].astype(str)

    print("Data set contains %d samples with %d features" % x.shape)

    if task == 'classification':
        original_silhouette = metrics.silhouette_score(x, y, metric='euclidean')
    elif task == 'regression':
        original_silhouette = 0.0

    file_to_save = dataset_file

    ax = utilstsne.plot(x, y, task=task, class_label=class_label, colors=COL, title='', draw_legend=False, draw_centers=False, draw_cluster_labels=False, s=dot_size)
    plt.savefig(output_folder + os.path.basename(file_to_save).replace('.csv','_act{}d{}'.format(2, '.pdf')), bbox_inches='tight')

    plotly_fig = utilstsne.iteractive_plot(df, x_label='heptocarcinoma', y_label='normal', z_label=None, task=task, sorted_classes={class_label: classes}, color_label=class_label, colors=[colhex[c] for c in class_colors], size=dot_size)
    plotly_fig.write_html(output_folder + os.path.basename(file_to_save).replace('.csv','_act{}d.html'.format(2)))

    plt.close()

if __name__ == '__main__': 
    main()

