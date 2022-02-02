# Bruno Iochins Grisci
# November 16th, 2021

dataset_file   = "DATA/juliano/LRRN_j.csv" # path to data file (must be .csv, features as columns, first row and first column are labels, first column after labels should contain the classes or target values)
weights_file   = "DATA/selections/LRRN/RelAgg_Solid_Tissue_Normal_LRRN_j.csv"
#'MRMR' 'MutualInformationFilter' 'KruskallWallisFilter''ReliefFFeatureSelector' 'Lasso' 'LinearSVM' 'SVMRFE''ReliefFGeneticAlgorithm' 'DecisionTree' 'RandomForest''SVMGeneticAlgorithm'
task           = "classification" # "classification" or "regression"
class_label    = "class"              # label of the column with the classes or target values
row_index      = 0           # The column that has the row index, None if no index
dataset_sep    = ","              # use ',' to separate columns in the dataset
output_folder  = 'JULIANO/LRRN/'        # name of directory in which the results will be saved

standardized   = True # True if data should be normalized with the z-norm (M=0.0, std=1.0)
rescaled       = False # True if data should be scaled between 0 and 1

dot_size       = 80
class_colors   = ['ORANGE', 'GREEN', 'DRKBRW', 'YELLOW', 'CYAN', 'RED', 'BEIGE', 'BLUE', 'BLACK', 'SILVER'] # list of colors to be assigned to the classes in the data, options as below:
                                     # 'RED', 'BLUE', 'YELLOW', 'GREEN', 'ORANGE', 'BLACK', 'CYAN', 'SILVER', 'MAGENTA', 'CREAM', 'DRKBRW', 'BEIGE', 'WHITE'
compute_pca    = True   # if PCA should be computed and shown
show_figs      = False   # if the plots should be displayed in addition to saved

no_weights     = False  # If True plots regular t-SNE
n_components   = 2      # dimensions of the embedding
perplexity     = 'auto' # perplexity value for t-SNE, if 'auto' then perplexity = max(30, x.shape[0]/100)
n_iter         = 500    # number of iterations for t-SNE