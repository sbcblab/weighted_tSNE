# Bruno Iochins Grisci
# October 16st, 2021

dataset_file   = "DATA/datasets/mnist_test.csv" # path to data file (must be .csv, features as columns, first row and first column are labels, first column after labels should contain the classes or target values)
weights_file   = "DATA/selections/RelAgg_mnist_test.csv"
#'MRMR' 'MutualInformationFilter' 'KruskallWallisFilter''ReliefFFeatureSelector' 'Lasso' 'LinearSVM' 'SVMRFE''ReliefFGeneticAlgorithm' 'DecisionTree' 'RandomForest''SVMGeneticAlgorithm'
task           = "classification" # "classification" or "regression"
class_label    = "label"              # label of the column with the classes or target values
row_index      = None           # The column that has the row index, None if no index
dataset_sep    = ","              # use ',' to separate columns in the dataset
output_folder  = 'RESULTS/test/'        # name of directory in which the results will be saved

standardized   = False # True if data should be normalized with the z-norm (M=0.0, std=1.0)
rescaled       = True # True if data should be scaled between 0 and 1

dot_size       = 40
class_colors   = ['RED', 'BLUE', 'YELLOW', 'CYAN', 'GREEN', 'BEIGE', 'DRKBRW', 'BLACK', 'SILVER', 'ORANGE'] # list of colors to be assigned to the classes in the data, options as below:
                                     # 'RED', 'BLUE', 'YELLOW', 'GREEN', 'ORANGE', 'BLACK', 'CYAN', 'SILVER', 'MAGENTA', 'CREAM', 'DRKBRW', 'BEIGE', 'WHITE'
compute_pca    = True   # if PCA should be computed and shown
show_figs      = False   # if the plots should be displayed in addition to saved

no_weights     = True  # If True plots regular t-SNE
n_components   = 3      # dimensions of the embedding
perplexity     = 'auto' # perplexity value for t-SNE, if 'auto' then perplexity = max(30, x.shape[0]/100)
n_iter         = 500    # number of iterations for t-SNE