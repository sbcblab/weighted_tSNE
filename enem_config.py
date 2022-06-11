# Bruno Iochins Grisci
# June 2nd, 2022

dataset_file   = "../RelAgg/DATA/enem/enem2016.csv" # path to data file (must be .csv, features as columns, first row and first column are labels, first column after labels should contain the classes or target values)
#dataset_file   = "IEEEVIS/DATA/Liver_GSE22405.csv" # path to data file (must be .csv, features as columns, first row and first column are labels, first column after labels should contain the classes or target values)
weights_file   = ["DATA/selections/movie/RelAgg_1000_score_enem2016.csv",  # list with paths to the files with the weights of each feature
                  ] 
#weights_file   = ["IEEEVIS/selections/LinearSVM_Liver.csv"]  
#weights_file   = ["IEEEVIS/selections/Perfect_Synth_A.csv"]         
# DecisionTree, KruskallWallisFilter, Lasso, MRMR, MutualInformationFilter, RandomForest, ReliefFFeatureSelector, LinearSVM, RelAgg    
# xor, Synth_A, Prostate, Liver
# Liver_GSE22405, Prostate_GSE6919_U95C, synth_100samples_5000features_50informative, xor_500samples_50features, regression_4in100_1000

task           = "regression" # "classification" or "regression"
class_label    = "NU_NOTA_MT"          # label of the column with the classes or target values
row_index      = 0             # The column that has the row index, None if no index
dataset_sep    = ","              # use ',' to separate columns in the dataset
output_folder  = 'RESULTS/movie/'  # name of directory in which the results will be saved
fig_extension  = '.png'

standardized   = False # True if data should be normalized with the z-norm (M=0.0, std=1.0)
rescaled       = True # True if data should be scaled between 0 and 1

title          = False # if True, print title, if False not
draw_centers   = True # if True plot the centers of each cluster
draw_legend    = False # if True plot the legend in the plot
dot_size       = 100    # size of dots in the plots
class_colors   = ['GREEN', 'ORANGE', 'RED', 'BLUE', 'YELLOW', 'CYAN', 'BEIGE', 'DRKBRW', 'BLACK', 'SILVER'] # list of colors to be assigned to the classes in the data, options as below:
                # 'RED', 'BLUE', 'YELLOW', 'GREEN', 'ORANGE', 'BLACK', 'CYAN', 'SILVER', 'MAGENTA', 'CREAM', 'DRKBRW', 'BEIGE', 'WHITE'
compute_pca    = False   # if PCA should be computed and shown
show_figs      = False  # if the plots should be displayed in addition to saved
rotation       = False   # if True rotates the t-SNE plot so that all plots have the same orientation (does nothing for PCA or 3D plots)

no_weights     = False  # If True plots regular t-SNE, if False plots weighted t-SNE using the values from weights_file
n_components   = 2      # dimensions of the embedding (1, 2, or 3)
perplexity     = 'auto' # perplexity value for t-SNE, if 'auto' then perplexity = max(30, x.shape[0]/100)
n_iter         = 500    # number of iterations for t-SNE
n_jobs         = 8  # Number of jobs for running t-SNE
random_seed    = 42 # Random seed for the t-SNE
initialization = "pca" # See options from openTSNE
metric         = 'euclidean' # See options from openTSNE
neighbors      = "auto" # See options from openTSNE
learning_rate  = "auto" # See options from openTSNE