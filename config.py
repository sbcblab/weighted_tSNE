# Bruno Iochins Grisci
# February 21st, 2022

dataset_file   = "DMDK/DATA/xor_500samples_50features.csv" # path to data file (must be .csv, features as columns, first row and first column are labels, first column after labels should contain the classes or target values)
#dataset_file   = "IEEEVIS/DATA/Liver_GSE22405.csv" # path to data file (must be .csv, features as columns, first row and first column are labels, first column after labels should contain the classes or target values)
weights_file   = ["../RelAgg/RESULTS/DMDK/xor_2in50_500/RelAgg_0015_score_xor_2in50_500.csv",
                  "../RelAgg/RESULTS/DMDK/xor_2in50_500/RelAgg_0030_score_xor_2in50_500.csv", 
                  "../RelAgg/RESULTS/DMDK/xor_2in50_500/RelAgg_0045_score_xor_2in50_500.csv",
                  "../RelAgg/RESULTS/DMDK/xor_2in50_500/RelAgg_0060_score_xor_2in50_500.csv",
                  "../RelAgg/RESULTS/DMDK/xor_2in50_500/RelAgg_0075_score_xor_2in50_500.csv",
                  "../RelAgg/RESULTS/DMDK/xor_2in50_500/RelAgg_0090_score_xor_2in50_500.csv",
                  "../RelAgg/RESULTS/DMDK/xor_2in50_500/RelAgg_0105_score_xor_2in50_500.csv",
                  "../RelAgg/RESULTS/DMDK/xor_2in50_500/RelAgg_0120_score_xor_2in50_500.csv",
                  "../RelAgg/RESULTS/DMDK/xor_2in50_500/RelAgg_0135_score_xor_2in50_500.csv",
                  "../RelAgg/RESULTS/DMDK/xor_2in50_500/RelAgg_0150_score_xor_2in50_500.csv",
                 ]
#weights_file   = ["IEEEVIS/selections/KruskallWallisFilter_xor.csv",  # list with paths to the files with the weights of each feature
#                  "IEEEVIS/selections/MutualInformationFilter_xor.csv",
#                  "IEEEVIS/selections/MRMR_xor.csv",
#                  "IEEEVIS/selections/ReliefFFeatureSelector_xor.csv",
#                  "IEEEVIS/selections/Lasso_xor.csv",
#                  "IEEEVIS/selections/DecisionTree_xor.csv",
#                  "IEEEVIS/selections/RandomForest_xor.csv",
#                  "IEEEVIS/selections/LinearSVM_xor.csv",
#                  "IEEEVIS/selections/RelAgg_xor.csv",] 
#weights_file   = ["IEEEVIS/selections/LinearSVM_Liver.csv"]  
#weights_file   = ["IEEEVIS/selections/Perfect_Synth_A.csv"]         
# DecisionTree, KruskallWallisFilter, Lasso, MRMR, MutualInformationFilter, RandomForest, ReliefFFeatureSelector, LinearSVM, RelAgg    
# xor, Synth_A, Prostate, Liver
# Liver_GSE22405, Prostate_GSE6919_U95C, synth_100samples_5000features_50informative, xor_500samples_50features, regression_4in100_1000

task           = "classification" # "classification" or "regression"
class_label    = "class"          # label of the column with the classes or target values
row_index      = None             # The column that has the row index, None if no index
dataset_sep    = ","              # use ',' to separate columns in the dataset
output_folder  = 'RESULTS/DMDK/xor/'  # name of directory in which the results will be saved
fig_extension  = '.pdf'

standardized   = False # True if data should be normalized with the z-norm (M=0.0, std=1.0)
rescaled       = False # True if data should be scaled between 0 and 1

title          = False # if True, print title, if False not
draw_centers   = False # if True plot the centers of each cluster
draw_legend    = False # if True plot the legend in the plot
dot_size       = 100    # size of dots in the plots
class_colors   = ['RED', 'BLUE', 'YELLOW', 'CYAN', 'GREEN', 'BEIGE', 'DRKBRW', 'BLACK', 'SILVER', 'ORANGE'] # list of colors to be assigned to the classes in the data, options as below:
                # 'RED', 'BLUE', 'YELLOW', 'GREEN', 'ORANGE', 'BLACK', 'CYAN', 'SILVER', 'MAGENTA', 'CREAM', 'DRKBRW', 'BEIGE', 'WHITE'
compute_pca    = False   # if PCA should be computed and shown
show_figs      = False  # if the plots should be displayed in addition to saved
rotation       = True   # if True rotates the t-SNE plot so that all plots have the same orientation (does nothing for PCA or 3D plots)

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