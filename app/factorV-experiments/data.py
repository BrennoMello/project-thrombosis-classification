import pandas as pd
import numpy as np

def read_dataset_v():
    df = pd.read_csv("data/FV_final_dataset_clean.csv", header=0)
    
    Y = df['mutation_observed'].copy()
    
    df.drop(['mutation_observed'], inplace=True, axis=1)
    X = df
    
    return X, Y 
