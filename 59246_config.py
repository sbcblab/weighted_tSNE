# Bruno Iochins Grisci
# June 2nd, 2022

dataset_file   = "DATA/Breast_GSE59246.csv" # path to data file (must be .csv, features as columns, first row and first column are labels, first column after labels should contain the classes or target values)
#dataset_file   = "IEEEVIS/DATA/Liver_GSE22405.csv" # path to data file (must be .csv, features as columns, first row and first column are labels, first column after labels should contain the classes or target values)
weights_file   = ["DATA/selections/movie/RelAgg_0001_score_Breast_GSE59246.csv",  # list with paths to the files with the weights of each feature
                  "DATA/selections/movie/RelAgg_0002_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0003_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0004_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0005_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0006_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0007_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0008_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0009_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0010_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0011_score_Breast_GSE59246.csv",  # list with paths to the files with the weights of each feature
                  "DATA/selections/movie/RelAgg_0012_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0013_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0014_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0015_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0016_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0017_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0018_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0019_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0020_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0021_score_Breast_GSE59246.csv",     
                  "DATA/selections/movie/RelAgg_0022_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0023_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0024_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0025_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0026_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0027_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0028_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0029_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0030_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0031_score_Breast_GSE59246.csv",  # list with paths to the files with the weights of each feature
                  "DATA/selections/movie/RelAgg_0032_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0033_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0034_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0035_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0036_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0037_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0038_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0039_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0040_score_Breast_GSE59246.csv",  
                  "DATA/selections/movie/RelAgg_0041_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0042_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0043_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0044_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0045_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0046_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0047_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0048_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0049_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0050_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0051_score_Breast_GSE59246.csv",  # list with paths to the files with the weights of each feature
                  "DATA/selections/movie/RelAgg_0052_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0053_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0054_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0055_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0056_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0057_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0058_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0059_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0060_score_Breast_GSE59246.csv", 
                  "DATA/selections/movie/RelAgg_0061_score_Breast_GSE59246.csv", 
                  "DATA/selections/movie/RelAgg_0062_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0063_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0064_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0065_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0066_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0067_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0068_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0069_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0070_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0071_score_Breast_GSE59246.csv",  # list with paths to the files with the weights of each feature
                  "DATA/selections/movie/RelAgg_0072_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0073_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0074_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0075_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0076_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0077_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0078_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0079_score_Breast_GSE59246.csv",
                  "DATA/selections/movie/RelAgg_0080_score_Breast_GSE59246.csv", 
                  ] 
#weights_file   = ["IEEEVIS/selections/LinearSVM_Liver.csv"]  
#weights_file   = ["IEEEVIS/selections/Perfect_Synth_A.csv"]         
# DecisionTree, KruskallWallisFilter, Lasso, MRMR, MutualInformationFilter, RandomForest, ReliefFFeatureSelector, LinearSVM, RelAgg    
# xor, Synth_A, Prostate, Liver
# Liver_GSE22405, Prostate_GSE6919_U95C, synth_100samples_5000features_50informative, xor_500samples_50features, regression_4in100_1000

task           = "classification" # "classification" or "regression"
class_label    = "type"          # label of the column with the classes or target values
row_index      = 0             # The column that has the row index, None if no index
dataset_sep    = ","              # use ',' to separate columns in the dataset
output_folder  = 'RESULTS/BREAST/'  # name of directory in which the results will be saved
fig_extension  = '.png'

standardized   = True # True if data should be normalized with the z-norm (M=0.0, std=1.0)
rescaled       = False # True if data should be scaled between 0 and 1

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