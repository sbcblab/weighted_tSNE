import numpy as np
import pandas as pd

def main():

    folder = 'TESE/parser/'

    datasets = [('xor_500samples_50features', 'xor_500samples_50features'),
                ('synth_100samples_5000features_50informative', 'synth_100samples_5000features_50informative'),
                ('Liver_GSE22405', 'Liver_GSE22405'),
                ('Prostate_GSE6919_U95C', 'Prostate_GSE6919_U95C')]
    metrics = ['mean', 'std']
    selectors = ['NoSelection', 'DecisionTree', 'KruskallWallisFilter', 'Lasso', 'LinearSVM', 
             'MRMR', 'MutualInformationFilter', 'RandomForest', 'ReliefFFeatureSelector',
             'ReliefFGeneticAlgorithm', 'SVMGeneticAlgorithm', 'SVMRFE', 'RelAgg']    

    metrics  = ['mean']
    datasets = [('xor_500samples_50features', 'xor_500samples_50features')]

    visualizers = ['tsne', 'pca']
    dimensions = ['2d', '3d']
    repetitions = 1

    columns = []
    for d in datasets:
        for e in ['Original', '2dembedding', '3dembedding']:
            for m in metrics:
                columns.append('{}_{}_{}'.format(d[0],e,m))
    print(columns)

    ddd = {}
    for visualizer in visualizers:
        ddd[visualizer] = pd.DataFrame(index=selectors, columns=columns)
        print(ddd[visualizer])
        for dataset in datasets:
            for selector in selectors:
                for dimension in dimensions:
                    for repetition in range(repetitions):
                        if selector == 'NoSelection':
                            file_name = folder + dataset[0] + 'silhouette_' + visualizer + dimension + '.csv'
                        else:
                            file_name = folder + selector + '_' + dataset[1] + 'silhouette_' + visualizer + dimension + '.csv'
                        try:
                            df = pd.read_csv(file_name, delimiter=',', header=0, index_col=0)
                            #print(df)
                            ddd[visualizer]['{}_{}_{}'.format(dataset[0],'Original','mean')][selector] = df['Weighted silhouette'][0]
                            ddd[visualizer]['{}_{}embedding_{}'.format(dataset[0], dimension,'mean')][selector] = df['Embedding silhouette'][0]
                            print(file_name)
                        except FileNotFoundError:
                            print("File does not exist: " + file_name)
                            pass
    for k in ddd:
        print('\n')
        print(k)
        print(ddd[k])

if __name__ == '__main__': 
    main()