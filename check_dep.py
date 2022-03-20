# Bruno Iochins Grisci
# May 24th, 2020

try:
    import sys
except:
    print('Could not import sys')

if sys.version_info < (3, 0):
    sys.stdout.write("Sorry, requires Python 3.x, not Python 2.x\n")
    sys.exit(1)

########################################

try:
    import os
except:
    print('Could not import os')
try:
    import importlib
except:
    print('Could not import importlib')
try:
    import importlib.util
except:
    print('Could not import importlib.util')
try:
    from collections import namedtuple
except:
    print('Could not import namedtuple from collections')
try:
    import pickle
except:
    print('Could not import pickle')
try:
    import itertools
except:
    print('Could not import itertools')
try:
    import tempfile
except:
    print('Could not import tempfile')

try:
    import importlib.util as imp
except:
    print('Could not import importlib.util')
try:
    if imp.find_spec("cupy"):
        import cupy
        import cupy as np
except:
    print('Could not import cupy')

########################################

try:
    import matplotlib
except:
    print('Could not import matplotlib')

matplotlib.use('Agg')

try:
    import matplotlib.pyplot as plt
except:
    print('Could not import matplotlib.pyplot')

try:
    import plotly
except:
    print('Could not import plotly')    

########################################

try:
    import numpy as np
except:
    print('Could not import numpy')

try:
    import pandas as pd
except:
    print('Could not import pandas')

print('Pandas version must be >= 0.25.3, it is {}'.format(pd.__version__))

########################################

try:
    import scipy.stats as stats
except:
    print('Could not import scipy.stats')
try:
    from scipy.stats import gmean
except:
    print('Could not import gmean from scipy.stats')
try:
    from scipy.stats.mstats import rankdata
except:
    print('Could not import rankdata from scipy.stats.mstats')

########################################

try:
    import sklearn
except:
    print('Could not import sklearn')    
try:
    from sklearn import metrics
except:
    print('Could not import metrics from sklearn')
try:
    from sklearn import datasets
except:
    print('Could not import datasets from sklearn')
try:
    from sklearn.preprocessing import StandardScaler
except:
    print('Could not import StandardScaler from sklearn.preprocessing')
try:
    from sklearn.decomposition import PCA
except:
    print('Could not import PCA from sklearn.decomposition')
try:
    from sklearn.manifold import TSNE
except:
    print('Could not import TSNE from sklearn.manifold')
try:
    from sklearn.utils import class_weight
except:
    print('Could not import class_weight from sklearn.utils')
try:
    from sklearn.model_selection import StratifiedKFold, KFold
except:
    print('Could not import StratifiedKFold, KFold from sklearn.model_selection')    

########################################

try:
    import model_io
except:
    print('Could not import model_io')
try:
    import RR_utils
except:
    print('Could not import RR_utils')
try:
    import plot_pca
except:
    print('Could not import plot_pca')

print('###############')
print('Finished checking dependencies. Please install any missing library!')