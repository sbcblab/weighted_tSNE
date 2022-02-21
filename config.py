# Bruno Iochins Grisci
# February 21st, 2022

dataset_file   = "DATA/datasets/mnist_test.csv" # path to data file (must be .csv, features as columns, first row and first column are labels, first column after labels should contain the classes or target values)
weights_file   = "DATA/selections/RelAgg_mnist_test.csv" # path to the file with the weights of each feature
#'MRMR' 'MutualInformationFilter' 'KruskallWallisFilter''ReliefFFeatureSelector' 'Lasso' 'LinearSVM' 'SVMRFE''ReliefFGeneticAlgorithm' 'DecisionTree' 'RandomForest''SVMGeneticAlgorithm'
task           = "classification" # "classification" or "regression"
class_label    = "label"          # label of the column with the classes or target values
row_index      = None             # The column that has the row index, None if no index
dataset_sep    = ","              # use ',' to separate columns in the dataset
output_folder  = 'RESULTS/test/'  # name of directory in which the results will be saved

standardized   = False # True if data should be normalized with the z-norm (M=0.0, std=1.0)
rescaled       = True # True if data should be scaled between 0 and 1

dot_size       = 40 # size of dots in the plots
class_colors   = ['RED', 'BLUE', 'YELLOW', 'CYAN', 'GREEN', 'BEIGE', 'DRKBRW', 'BLACK', 'SILVER', 'ORANGE'] # list of colors to be assigned to the classes in the data, options as below:
                # 'RED', 'BLUE', 'YELLOW', 'GREEN', 'ORANGE', 'BLACK', 'CYAN', 'SILVER', 'MAGENTA', 'CREAM', 'DRKBRW', 'BEIGE', 'WHITE'
compute_pca    = True   # if PCA should be computed and shown
show_figs      = False  # if the plots should be displayed in addition to saved
rotation       = True   # if True rotates the t-SNE plot so that all plots have the same orientation (does nothing for PCA or 3D plots)

no_weights     = True   # If True plots regular t-SNE, if False plots weighted t-SNE using the values from weights_file
n_components   = 2      # dimensions of the embedding (1, 2, or 3)
perplexity     = 'auto' # perplexity value for t-SNE, if 'auto' then perplexity = max(30, x.shape[0]/100)
n_iter         = 500    # number of iterations for t-SNE