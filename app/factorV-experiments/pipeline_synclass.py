import sys
sys.path.append('../')
import time
import pickle
import warnings
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from loguru import logger as log
from datetime import datetime
from collections import Counter
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from synclass.sfc import ModelSearchSynClass
from synclass.sfc import SynClass

from ialovecoffe.data import *
from ialovecoffe.models import *
from ialovecoffe.validation import computer_scores, computer_scores_outlier, accuracy_per_class
from sklearn.utils import shuffle
from sklearn.metrics import RocCurveDisplay

# config
warnings.simplefilter("ignore")
random.seed(10)

def undersampling(X, Y, percentage, rs, at='target', increase=3, sm=False):
    
    X[at] = Y
    
    # surffle
    X = shuffle(X, random_state=rs)

    #size_minority = min(Counter(X[at]).values())
    proportions = Counter(X[at])

    class_minority = min(proportions, key=proportions.get)
    size_minority  = proportions[class_minority]
    
    p = np.ceil(size_minority * percentage).astype('int')
    p_train = (size_minority - p)
        
    train, test = [], []

    for classe in X[at].unique():
        
        df_class = X[X[at] == classe]

        if classe != class_minority:
            train.append(df_class.iloc[p:(p_train*increase)])
        else:
            train.append(df_class.iloc[p:(p_train)])        
            
        test.append(df_class.iloc[:p])
        #train.append(df_class.iloc[p:p_train])
        
    df_train = pd.concat(train)
    df_test = pd.concat(test)
    
    y_train = df_train[at]
    y_test = df_test[at]
        
    x_train = df_train.drop([at], axis=1)
    x_test = df_test.drop([at], axis=1)   

    if sm:
        x_train, y_train = SMOTE().fit_resample(x_train, y_train)
    
    return x_train, y_train, x_test, y_test



def test_balacing(X, Y, percentage, rs, at='target', sm=False):
    
    X[at] = Y

    # surffle
    X = shuffle(X, random_state=rs)

    proportions = Counter(X[at])

    class_minority = min(proportions, key=proportions.get)
    size_minority  = proportions[class_minority]
    
    p = np.ceil(size_minority * percentage).astype('int')
    p_train = (size_minority - p)
        
    train, test = [], []

    for classe in X[at].unique():
        
        df_class = X[X[at] == classe]
        
        test.append(df_class.iloc[:p])
        train.append(df_class.iloc[p:])
        
    df_train = pd.concat(train)
    df_test = pd.concat(test)
    
    y_train = df_train[at]
    y_test = df_test[at]
        
    x_train = df_train.drop([at], axis=1)
    x_test = df_test.drop([at], axis=1)   

    if sm:
        x_train, y_train = SMOTE().fit_resample(x_train, y_train)
    
    return x_train, y_train, x_test, y_test


def roc_calc_viz(classifier, X_test, y_test):
    viz = RocCurveDisplay.from_estimator(
                classifier,
                X_test,
                y_test
            )
    
    return viz.fpr, viz.tpr, viz.roc_auc

def roc_calc_viz_pred(y_true, y_pred):
    viz = RocCurveDisplay.from_predictions(
                            y_true,
                            y_pred
                        )

    return viz.fpr, viz.tpr, viz.roc_auc


def run_experiment(x, y, iterations, p, dsetname) -> pd.DataFrame:

    data_results = []

    #results = {'model_name': [], 'iteration':[], 'F1':[], 
    #            'ROC':[],'acc-class-1':[],'acc-class-2':[], 'SEN':[], 
    #            'SPE':[], 'MCC':[], 'TPR': [], 'FPR':[], 'AUC': [], 'THRE': []}

    final_results = {'model_name': [], 'F1':[],  'ROC':[], 'acc-class-1':[], 'acc-class-2':[], 
                    'TPR': [], 'FPR':[], 'AUC': [], 'THRE': []}

    for i in tqdm(range(iterations)):

        x_train_raw, y_train_raw, x_test_raw, y_test_raw = test_balacing(x, y, p, i, True)

        #x_train = x_train_raw.to_numpy()
        #x_test = x_test_raw.to_numpy()
        y_train = y_train_raw.to_numpy()
        y_test = y_test_raw.to_numpy()

        log.debug('-' * 30)
        log.debug(f'{dsetname} - Iteration {i} - test size: {p}')
        log.debug('-' * 30)

        results_iter = ModelSearchSynClass(x_train_raw, y_train_raw, x_test_raw, y_test_raw, learner = RSLocalOutlierFactor , scoring = 'f1_macro', 
                                          classThreshold = 0.5, probability = False, rangeThreshold = [-0.1, 0.81, 0.01], results = final_results)


    df_fold = pd.DataFrame(final_results)
    models = df_fold['model_name'].unique()
    log.info('\n')
    log.info('-' * 30)
    for model in models:

        df_model = df_fold[df_fold['model_name'] == model]
        mean_f1 = float(f'{np.mean(df_model["F1"]):.2f}')

        log.info(f'MODEL {model} with .....: {mean_f1}')

    return df_fold
 






