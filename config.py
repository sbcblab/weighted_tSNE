# Bruno Iochins Grisci
# February 21st, 2022

dataset_file   = "IEEEVIS/DATA/xor_500samples_50features.csv" # path to data file (must be .csv, features as columns, first row and first column are labels, first column after labels should contain the classes or target values)
#dataset_file   = "IEEEVIS/DATA/Liver_GSE22405.csv" # path to data file (must be .csv, features as columns, first row and first column are labels, first column after labels should contain the classes or target values)
weights_file   = ["DATA/selections/movie/xor/RelAgg_0001_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0002_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0003_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0004_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0005_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0006_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0007_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0008_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0009_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0010_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0011_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0012_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0013_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0014_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0015_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0016_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0017_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0018_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0019_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0020_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0021_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0022_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0023_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0024_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0025_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0026_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0027_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0028_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0029_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0030_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0031_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0032_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0033_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0034_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0035_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0036_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0037_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0038_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0039_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0030_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0041_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0042_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0043_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0044_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0045_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0046_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0047_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0048_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0049_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0050_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0051_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0052_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0053_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0054_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0055_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0056_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0057_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0058_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0059_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0060_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0061_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0062_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0063_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0064_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0065_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0066_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0067_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0068_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0069_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0070_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0071_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0072_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0073_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0074_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0075_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0076_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0077_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0078_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0079_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0080_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0081_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0082_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0083_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0084_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0085_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0086_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0087_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0088_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0089_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0090_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0081_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0082_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0083_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0084_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0085_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0086_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0087_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0088_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0089_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0090_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0081_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0082_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0083_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0084_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0085_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0086_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0087_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0088_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0089_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0090_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0091_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0092_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0093_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0094_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0095_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0096_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0097_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0098_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0099_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0110_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0111_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0112_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0113_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0114_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0115_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0116_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0117_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0118_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0119_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0120_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0121_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0122_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0123_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0124_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0125_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0126_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0127_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0128_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0129_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0130_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0131_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0132_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0133_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0134_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0135_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0136_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0137_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0138_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0139_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0141_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0142_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0143_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0144_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0145_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0146_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0147_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0148_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0149_score_xor_2in50_500.csv",
                    "DATA/selections/movie/xor/RelAgg_0150_score_xor_2in50_500.csv",                                                        
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
output_folder  = 'RESULTS/movie/xor/'  # name of directory in which the results will be saved
fig_extension  = '.png'

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