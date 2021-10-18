# Bruno Iochins Grisci
# February 18st, 2021

dataset_file   = "DATA/datasets/xor_500samples_50features.csv" # path to data file (must be .csv, features as columns, first row and first column are labels, first column after labels should contain the classes or target values)
weights_file   = "DATA/selections/RelAgg_xor_num_features_50_num_selected_50.csv"
#'MRMR' 'MutualInformationFilter' 'KruskallWallisFilter''ReliefFFeatureSelector' 'Lasso' 'LinearSVM' 'SVMRFE''ReliefFGeneticAlgorithm' 'DecisionTree' 'RandomForest''SVMGeneticAlgorithm'
task           = "classification" # "classification" or "regression"
class_label    = "class"              # label of the column with the classes or target values
row_index      = None           # The column that has the row index, None if no index
dataset_sep    = ","              # use ',' to separate columns in the dataset
output_folder  = 'TESE/XOR/'        # name of directory in which the results will be saved

standardized   = True # True if data should be normalized with the z-norm (M=0.0, std=1.0)
rescaled       = False # True if data should be scaled between 0 and 1

dot_size       = 80
class_colors   = ['RED', 'BLUE', 'ORANGE', 'GREEN', 'YELLOW', 'BLACK'] # list of colors to be assigned to the classes in the data, options as below:
                                     # 'RED', 'BLUE', 'YELLOW', 'GREEN', 'ORANGE', 'BLACK', 'CYAN', 'SILVER', 'MAGENTA', 'CREAM', 'DRKBRW', 'BEIGE', 'WHITE'
compute_pca    = True   # if PCA should be computed and shown
show_figs      = True   # if the plots should be displayed in addition to saved

no_weights     = False  # If True plots regular t-SNE
n_components   = 2      # dimensions of the embedding
perplexity     = 'auto' # perplexity value for t-SNE, if 'auto' then perplexity = max(30, x.shape[0]/100)
n_iter         = 500    # number of iterations for t-SNE