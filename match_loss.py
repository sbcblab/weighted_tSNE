# Bruno Iochins Grisci
# June 28th, 2022

import os
import sys
import numpy as np
import pandas as pd

def main():

    loss_file = sys.argv[1]
    silh_file = sys.argv[2]
    freq      = int(sys.argv[3])
    
    df_loss = pd.read_csv(loss_file, delimiter=',', header=0, index_col=None)
    df_silh = pd.read_csv(silh_file, delimiter=',', header=0, index_col=0)
    
    df_loss['epoch'] += 1
    
    print(df_loss)
    print(df_silh)
    
    df_loss = df_loss[df_loss['epoch'] % freq == 0]
    
    df_loss.reset_index(drop=True, inplace=True)
    df_silh.reset_index(drop=True, inplace=True)
    
    df_stack = pd.concat([df_loss, df_silh], axis=1)
    print(df_stack)
    df_stack.to_csv(silh_file.replace('.csv', 'Xloss.csv'))
    
    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html
    for col1 in ['loss', 'accuracy', 'KL divergence']:
        for col2 in ["Embedding silhouette", 'Weighted silhouette', 'KL divergence']:
            corr = df_stack[col1].corr(df_stack[col2])
            print ("Correlation between ", col1, " and ", col2, "is: ", round(corr, 2))    
        
if __name__ == '__main__': 
    main()