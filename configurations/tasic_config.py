# Bruno Iochins Grisci
# June 2nd, 2022

import numpy as np

dataset_file   = "DATA/tasic_filt.csv" # path to data file (must be .csv, features as columns, first row and first column are labels, first column after labels should contain the classes or target values)
#dataset_file   = "IEEEVIS/DATA/Liver_GSE22405.csv" # path to data file (must be .csv, features as columns, first row and first column are labels, first column after labels should contain the classes or target values)
#weights_file   = ["DATA/selections/RF_score_tasic_filt2.csv", "DATA/selections/Lasso_score_tasic_filt2.csv"]  # list with paths to the files with the weights of each feature
weights_file   = [
                  #"DATA/selections/tasic/RelAgg_0007_score_tasic_filt.csv",
                  #"DATA/selections/tasic/RelAgg_0008_score_tasic_filt.csv",
                  #"DATA/selections/tasic/RelAgg_0009_score_tasic_filt.csv",
                  "DATA/selections/tasic/RelAgg_0010_score_tasic_filt.csv",
                 ]         
# DecisionTree, KruskallWallisFilter, Lasso, MRMR, MutualInformationFilter, RandomForest, ReliefFFeatureSelector, LinearSVM, RelAgg    
# xor, Synth_A, Prostate, Liver
# Liver_GSE22405, Prostate_GSE6919_U95C, synth_100samples_5000features_50informative, xor_500samples_50features, regression_4in100_1000
output_folder  = 'RESULTS/tasic2/'  # name of directory in which the results will be saved
fig_extension  = '.pdf'

task           = "classification" # "classification" or "regression"
class_label    = "subclass"          # label of the column with the classes or target values
row_index      = 0             # The column that has the row index, None if no index
load_chunksize  = 1000             # Chunksize used to load the dataset
dtype_float     = np.float32       # default dtype for float
dtype_int       = np.int32          # default dtype for integer
# https://vincentteyssier.medium.com/optimizing-the-size-of-a-pandas-dataframe-for-low-memory-environment-5f07db3d72e
# int8 / uint8 : consumes 1 byte of memory, range between -128/127 or 0/255
# float16 / int16 / uint16: consumes 2 bytes of memory, range between -32768 and 32767 or 0/65535
# float32 / int32 / uint32 : consumes 4 bytes of memory, range between -2147483648 and 2147483647
# float64 / int64 / uint64: consumes 8 bytes of memory
dataset_sep    = ","              # use ',' to separate columns in the dataset

standardized   = True # True if data should be normalized with the z-norm (M=0.0, std=1.0)
rescaled       = False # True if data should be scaled between 0 and 1

title               = False # if True, print title, if False not
draw_centers        = True # if True plot the centers of each cluster
draw_cluster_labels = True # if True plot the class name above the centers of each cluster
draw_legend         = False # if True plot the legend in the plot
pos_legend          = 'center left'
# The strings 'upper left', 'upper right', 'lower left', 'lower right' place the legend at the corresponding corner of the axes/figure.
# The strings 'upper center', 'lower center', 'center left', 'center right' place the legend at the center of the corresponding edge of the axes/figure.
# The string 'center' places the legend at the center of the axes/figure.
# The string 'best' places the legend at the location, among the nine locations defined so far, with the minimum overlap with other drawn artists. This option can be quite slow for plots with large amounts of data; your plotting speed may benefit from providing a specific location.
dot_size       = 7    # size of dots in the plots
#class_colors   = ['GREEN', 'ORANGE', 'RED', 'BLUE', 'YELLOW', 'CYAN', 'BEIGE', 'DRKBRW', 'BLACK', 'SILVER'] 
# list of colors to be assigned to the classes in the data, options as below:
# 'RED', 'BLUE', 'YELLOW', 'GREEN', 'ORANGE', 'BLACK', 'CYAN', 'SILVER', 'MAGENTA', 'CREAM', 'DRKBRW', 'BEIGE', 'WHITE'
# or pass a list of hexcolor strings such as '#9189FF', '#FFA388', '#882E81'...
class_colors = ['#665C47',
                '#00FF66',
                '#997F7A',
                '#A6E6A9',
                '#00979D',
                '#00DDC5',
                '#29E043',
                '#578EBF',
                '#A19922',
                '#1F6666',
                '#FFA388',
                '#598069',
                '#FF0000',
                '#3D9946',
                '#6B998D',
                '#665547',
                '#AF3F64',
                '#807059',
                '#CA66FF',
                '#6C00BF',
                '#B06411',
                '#89997A',
                '#9189FF']  

compute_pca    = True   # if PCA should be computed and shown
show_figs      = False  # if the plots should be displayed in addition to saved
rotation       = True   # if True rotates the t-SNE plot so that all plots have the same orientation (does nothing for PCA or 3D plots)

no_weights     = False  # If True plots regular t-SNE, if False plots weighted t-SNE using the values from weights_file
n_components   = 2      # dimensions of the embedding (1, 2, or 3)
perplexity     = 'auto' # perplexity value for t-SNE, if 'auto' then perplexity = max(30, x.shape[0]/100)
n_iter         = 500    # number of iterations for t-SNE
n_jobs         = 16  # Number of jobs for running t-SNE
random_seed    = 42 # Random seed for the t-SNE
initialization = "pca" # See options from openTSNE
metric         = 'euclidean' # See options from openTSNE
neighbors      = "auto" # See options from openTSNE
learning_rate  = "auto" # See options from openTSNE
