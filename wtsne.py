# Bruno Iochins Grisci
# February 18st, 2021
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
from matplotlib.colors import is_color_like
import plotly.tools as tls
import plotly.express as px
import utilstsne
from tqdm import tqdm

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
    print('Writing file {}'.format(cfg.output_folder + os.path.basename(file_to_save).replace('.csv','_pca{}d.csv'.format(cfg.n_components))))
    pca_emb.to_csv(cfg.output_folder + os.path.basename(file_to_save).replace('.csv','_pca{}d.csv'.format(cfg.n_components)))
    print(pca_emb)

    zl = None
    if cfg.n_components > 2:
        zl = pca_columns_labels[2]    
    fig = utilstsne.iteractive_plot(pca_emb, x_label=pca_columns_labels[0], y_label=pca_columns_labels[1], z_label=zl, task=cfg.task, sorted_classes={class_label: classes}, color_label=cfg.class_label, colors=[colhex.get(c, c) for c in cfg.class_colors], size=cfg.dot_size)
    fig.write_html(cfg.output_folder + os.path.basename(file_to_save).replace('.csv','_pca{}d.html'.format(cfg.n_components)))
    if cfg.show_figs:
        fig.show()

    if cfg.title:
        plot_title = os.path.basename(file_to_save).replace('.csv','')
    else:
        plot_title = ''

    ax = utilstsne.plot(components, y, task=cfg.task, class_label=cfg.class_label, colors=colors, title=plot_title, draw_legend=cfg.draw_legend, pos_legend=cfg.pos_legend, draw_centers=cfg.draw_centers, draw_cluster_labels=cfg.draw_cluster_labels, s=cfg.dot_size)
    print('Saving file {}'.format(cfg.output_folder + os.path.basename(file_to_save).replace('.csv','_pca{}d{}'.format(cfg.n_components, cfg.fig_extension))))
    plt.savefig(cfg.output_folder + os.path.basename(file_to_save).replace('.csv','_pca{}d{}'.format(cfg.n_components, cfg.fig_extension)), bbox_inches='tight')

    if cfg.task == 'classification':
        embedding_silhouette = metrics.silhouette_score(components, y, metric='euclidean')
    elif cfg.task == 'regression':
        embedding_silhouette = 0.0
        
    silhouettes = {'Original silhouette': [original_silhouette], 'Weighted silhouette': [weighted_silhouette], 'Embedding silhouette': [embedding_silhouette]}
    s_df = pd.DataFrame.from_dict(silhouettes)
    print(s_df)
    print('Writing file {}'.format(cfg.output_folder + os.path.basename(file_to_save).replace('.csv','silhouette_pca{}d.csv'.format(cfg.n_components))))
    s_df.to_csv(cfg.output_folder + os.path.basename(file_to_save).replace('.csv','silhouette_pca{}d.csv'.format(cfg.n_components)))

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

    if cfg.task == 'classification':
        original_silhouette = metrics.silhouette_score(x, y, metric='euclidean')
    elif cfg.task == 'regression':
        try:
            y_reg = y.astype(float)
            y_med = y_reg.median()
            y_reg[y_reg >= y_med] = 1
            y_reg[y_reg <  y_med] = 0
            y_reg = y_reg.astype(str)
            print(cfg.class_label + ' masked to compute silhouette score in regression:')
            print(y_reg)
            original_silhouette = metrics.silhouette_score(x, y_reg, metric='euclidean')
        except:
            original_silhouette = 0.0

    if cfg.no_weights:
        cfg.weights_file = ['0']

    all_sel_sil = {'Selection': [], 'Original silhouette': [], 'Weighted silhouette': [], 'Embedding silhouette': [], 'KL divergence': []}    
       
    index_df = df.index
    
    for selector_file in cfg.weights_file:

        print('### {}'.format(selector_file))

        if cfg.no_weights:
            file_to_save = cfg.dataset_file
        else:
            file_to_save = selector_file

        if not cfg.no_weights:
            if dummy == False:
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

        if cfg.task == 'classification':
            weighted_silhouette = metrics.silhouette_score(wx, y, metric='euclidean')
        elif cfg.task == 'regression':
            try:
                weighted_silhouette = metrics.silhouette_score(wx, y_reg, metric='euclidean')
            except:
                weighted_silhouette = 0.0
                       
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
            initialization=cfg.initialization,
            metric=cfg.metric,
            neighbors=cfg.neighbors,
            learning_rate=cfg.learning_rate,
            negative_gradient_method=ngm,
            verbose=True,
            n_jobs=cfg.n_jobs,
            n_iter=cfg.n_iter,
            random_state=cfg.random_seed,
        )

        embedding = tsne.fit(wx)
        #print(embedding)
        divergence = embedding.kl_divergence
        
        if cfg.rotation and embedding.shape[1] == 2:
            emb_columns_labels = ['wt-SNE {}'.format(i+1) for i in range(cfg.n_components)]
            emb = pd.DataFrame(data=embedding,    # values
                               index=index_df,    # 1st column as index
                               columns=emb_columns_labels)  # 1st row as the column names
            emb[cfg.class_label] = y
            if cfg.task == 'classification':
                first_class = emb[emb[cfg.class_label] == CLASS_LABELS[0]]
            elif cfg.task == 'regression':
                first_class = emb[emb[cfg.class_label].astype(float) <= 0.0]
            print(len(first_class[first_class['wt-SNE 1'] >= 0.0]))
            coord1_pos = len(first_class[first_class['wt-SNE 1'] >= 0.0])
            if coord1_pos > 3:
                first_class = first_class[first_class['wt-SNE 1'] >= 0.0]
            rotation_point = [first_class['wt-SNE 1'].mean(), first_class['wt-SNE 2'].mean()]
            embedding = utilstsne.rotate(embedding, rotation_point)
            #print(embedding)

        emb_columns_labels = ['wt-SNE {}'.format(i+1) for i in range(cfg.n_components)]
        emb = pd.DataFrame(data=embedding,    # values
                           index=index_df,    # 1st column as index
                           columns=emb_columns_labels)  # 1st row as the column names
        emb[cfg.class_label] = y
        print('Writing file {}'.format(cfg.output_folder + os.path.basename(file_to_save).replace('.csv','_tsne{}d.csv'.format(cfg.n_components))))
        emb.to_csv(cfg.output_folder + os.path.basename(file_to_save).replace('.csv','_tsne{}d.csv'.format(cfg.n_components)))
        print(emb)

        if cfg.task == 'classification':
            embedding_silhouette = metrics.silhouette_score(embedding, y, metric='euclidean')
        elif cfg.task == 'regression':
            try:
                embedding_silhouette = metrics.silhouette_score(embedding, y_reg, metric='euclidean')
            except:           
                embedding_silhouette = 0.0

        if cfg.title:
            plot_title = os.path.basename(file_to_save).replace('.csv','')
        else:
            plot_title = ''

        ax = utilstsne.plot(embedding, y, task=cfg.task, class_label=cfg.class_label, colors=COL, title=plot_title, draw_legend=cfg.draw_legend, pos_legend=cfg.pos_legend, draw_centers=cfg.draw_centers, draw_cluster_labels=cfg.draw_cluster_labels, s=cfg.dot_size)
        print('Saving file {}'.format(cfg.output_folder + os.path.basename(file_to_save).replace('.csv','_tsne{}d{}'.format(cfg.n_components, cfg.fig_extension))))
        plt.savefig(cfg.output_folder + os.path.basename(file_to_save).replace('.csv','_tsne{}d{}'.format(cfg.n_components, cfg.fig_extension)), bbox_inches='tight')

        if cfg.n_components > 1:
            zl = None
            if cfg.n_components > 2:
                zl = emb_columns_labels[2]
            plotly_fig = utilstsne.iteractive_plot(emb, x_label=emb_columns_labels[0], y_label=emb_columns_labels[1], z_label=zl, task=cfg.task, sorted_classes={cfg.class_label: classes}, color_label=cfg.class_label, colors=[colhex.get(c, c) for c in cfg.class_colors], size=cfg.dot_size)
            plotly_fig.write_html(cfg.output_folder + os.path.basename(file_to_save).replace('.csv','_tsne{}d.html'.format(cfg.n_components)))
            if cfg.show_figs:
                plotly_fig.show()

        silhouettes = {'Original silhouette': [original_silhouette], 'Weighted silhouette': [weighted_silhouette], 'Embedding silhouette': [embedding_silhouette], 'KL divergence': [divergence]}
        s_df = pd.DataFrame.from_dict(silhouettes)
        print(s_df)
        print('Writing file {}'.format(cfg.output_folder + os.path.basename(file_to_save).replace('.csv','silhouette_tsne{}d.csv'.format(cfg.n_components))))
        s_df.to_csv(cfg.output_folder + os.path.basename(file_to_save).replace('.csv','silhouette_tsne{}d.csv'.format(cfg.n_components)))

        all_sel_sil['Selection'].append(selector_file) 
        all_sel_sil['Original silhouette'].append(original_silhouette)
        all_sel_sil['Weighted silhouette'].append(weighted_silhouette)
        all_sel_sil['Embedding silhouette'].append(embedding_silhouette)
        all_sel_sil['KL divergence'].append(divergence)
        
        plt.close()
    
    del df
    all_sel_sil_df = pd.DataFrame.from_dict(all_sel_sil)
    print(all_sel_sil_df)
    print('Writing file {}'.format(cfg.output_folder + 'selectors_silhouette_tsne{}d.csv'.format(cfg.n_components)))
    all_sel_sil_df.to_csv(cfg.output_folder + 'selectors_silhouette_tsne{}d.csv'.format(cfg.n_components))
        
if __name__ == '__main__': 
    main()

