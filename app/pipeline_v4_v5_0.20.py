import sys
import time
import pickle
import warnings
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from loguru import logger as log
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
from collections import Counter

from ialovecoffe.data import *
from ialovecoffe.models import *
from ialovecoffe.validation import computer_scores, computer_scores_outlier, accuracy_per_class
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

#simplefilter("ignore", category=[RuntimeWarning, ConvergenceWarning]) 
#simplefilter("ignore", category=ConvergenceWarning)
warnings.simplefilter("ignore")

random.seed(10)

def create_folds(X, Y, FOLDS: int):

    skf = StratifiedKFold(n_splits=FOLDS, random_state=None, shuffle=False)

    folds = []

    for train_index, test_index in skf.split(X, Y):

        x_train = X.iloc[train_index].to_numpy()
        x_test = X.iloc[test_index].to_numpy()

        y_train = Y[train_index]
        y_test = Y[test_index]
        
        folds.append([x_train, y_train, x_test, y_test])

    return folds

def create_balanced_folds(X, Y, percentage, rs, at='target'): 
    size_minority = min(Counter(Y).values())

    X[at] = Y
    
    # surffle
    X = shuffle(X, random_state=rs)

    p = np.ceil(size_minority * percentage).astype('int')
    train = []
    test = []
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
    
    return x_train, y_train, x_test, y_test

def run_experiment(x, y, iterations) -> pd.DataFrame:

    data_results = []

    results = {'model_name': [], 'iteration':[], 'F1':[], 
                'ROC':[],'acc-class-1':[],'acc-class-2':[], 'SEN':[], 
                'SPE':[], 'MCC':[], 'TPR': [], 'FPR':[], 'THRESH': []}

    for i in tqdm(range(iterations)):

        #folds = create_folds(x, y, 2)

        #for j, fold in enumerate(folds):
            
        #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

        x_train, y_train, x_test, y_test = create_balanced_folds(x, y, rs=i, percentage=0.20)
        print(len(y_train))
        print(len(y_test))

        x_train = x_train.to_numpy()
        x_test = x_test.to_numpy()
        y_train = y_train.to_numpy()
        y_test = y_test.to_numpy()

        log.debug('-' * 30)
        log.debug(f'Iteration {i}')
        log.debug('-' * 30)
            
        model, y_pred, y_pred_prob = RSRF(x_train, y_train, x_test, y_test)
        sen, spe, f1, roc, jac, fmi, mcc, fpr, tpr, thresh = computer_scores(y_test, y_pred, y_pred_prob)
        acc1, acc2 = accuracy_per_class(y_test, y_pred)
        results['model_name'].append('RandomForest')
        results['iteration'].append(i)
        results['acc-class-1'].append(acc1)
        results['acc-class-2'].append(acc2)
        results['F1'].append(f1)
        results['ROC'].append(roc)
        results['SEN'].append(sen)
        results['SPE'].append(spe)
        results['MCC'].append(mcc)
        results['FPR'].append(fpr)
        results['TPR'].append(tpr)
        results['THRESH'].append(thresh)
        log.debug(f'RF .......: {f1}')

        
        model, y_pred, y_pred_prob = RSDT(x_train, y_train, x_test, y_test)
        sen, spe, f1, roc, jac, fmi, mcc, fpr, tpr, thresh = computer_scores(y_test, y_pred, y_pred_prob)
        acc1, acc2 = accuracy_per_class(y_test, y_pred)
        results['model_name'].append('DecisionTree')
        results['iteration'].append(i)
        results['acc-class-1'].append(acc1)
        results['acc-class-2'].append(acc2)
        results['F1'].append(f1)
        results['ROC'].append(roc)
        results['SEN'].append(sen)
        results['SPE'].append(spe)
        results['MCC'].append(mcc)
        results['FPR'].append(fpr)
        results['TPR'].append(tpr)
        results['THRESH'].append(thresh)
        log.debug(f'DT .......: {f1}')

        model, y_pred, y_pred_prob = RSNN(x_train, y_train, x_test, y_test)
        sen, spe, f1, roc, jac, fmi, mcc, fpr, tpr, thresh = computer_scores(y_test, y_pred, y_pred_prob)
        acc1, acc2 = accuracy_per_class(y_test, y_pred)
        results['model_name'].append('KNN')
        results['iteration'].append(i)
        results['F1'].append(f1)
        results['acc-class-1'].append(acc1)
        results['acc-class-2'].append(acc2)
        results['ROC'].append(roc)
        results['SEN'].append(sen)
        results['SPE'].append(spe)
        results['MCC'].append(mcc)
        results['FPR'].append(fpr)
        results['TPR'].append(tpr)
        results['THRESH'].append(thresh)
        log.debug(f'KNN ......: {f1}')
            
        model, y_pred = RSOneClassSVM(x_train, y_train, x_test, y_test)
        sen, spe, f1, roc, jac, fmi, mcc, fpr, tpr, thresh = computer_scores_outlier(y_test, y_pred)
        acc1, acc2 = accuracy_per_class(y_test, y_pred)
        results['model_name'].append('OneClassSVM')
        results['iteration'].append(i)
        results['acc-class-1'].append(acc1)
        results['acc-class-2'].append(acc2)
        results['F1'].append(f1)
        results['ROC'].append(roc)
        results['SEN'].append(sen)
        results['SPE'].append(spe)
        results['MCC'].append(mcc)
        results['FPR'].append(fpr)
        results['TPR'].append(tpr)
        results['THRESH'].append(thresh)
        log.debug(f'OneClassSVM ......: {f1}')

        model, y_pred = RSLocalOutlierFactor(x_train, y_train, x_test, y_test)
        sen, spe, f1, roc, jac, fmi, mcc, fpr, tpr, thresh = computer_scores_outlier(y_test, y_pred)
        acc1, acc2 = accuracy_per_class(y_test, y_pred)
        results['model_name'].append('LocalOutlierFactor')
        results['iteration'].append(i)
        results['acc-class-1'].append(acc1)
        results['acc-class-2'].append(acc2)
        results['F1'].append(f1)
        results['ROC'].append(roc)
        results['SEN'].append(sen)
        results['SPE'].append(spe)
        results['MCC'].append(mcc)
        results['FPR'].append(fpr)
        results['TPR'].append(tpr)
        results['THRESH'].append(thresh)
        log.debug(f'LocalOutlierFactor ......: {f1}')

        model, y_pred = RSIsolationForest(x_train, y_train, x_test, y_test)
        sen, spe, f1, roc, jac, fmi, mcc, fpr, tpr, thresh = computer_scores_outlier(y_test, y_pred)
        acc1, acc2 = accuracy_per_class(y_test, y_pred)
        results['model_name'].append('IsolationForest')
        results['iteration'].append(i)
        results['acc-class-1'].append(acc1)
        results['acc-class-2'].append(acc2)
        results['F1'].append(f1)
        results['ROC'].append(roc)
        results['SEN'].append(sen)
        results['SPE'].append(spe)
        results['MCC'].append(mcc)
        results['FPR'].append(fpr)
        results['TPR'].append(tpr)
        results['THRESH'].append(thresh)
        log.debug(f'IsolationForest ......: {f1}')

        model, y_pred, y_pred_prob = RSXgboost(x_train, y_train, x_test, y_test)
        sen, spe, f1, roc, jac, fmi, mcc, fpr, tpr, thresh = computer_scores(y_test, y_pred, y_pred_prob)
        acc1, acc2 = accuracy_per_class(y_test, y_pred)
        results['model_name'].append('Xgboost')
        results['iteration'].append(i)
        results['acc-class-1'].append(acc1)
        results['acc-class-2'].append(acc2)
        results['F1'].append(f1)
        results['ROC'].append(roc)
        results['SEN'].append(sen)
        results['SPE'].append(spe)
        results['MCC'].append(mcc)
        results['FPR'].append(fpr)
        results['TPR'].append(tpr)
        results['THRESH'].append(thresh)
        log.debug(f'Xgboost ......: {f1}')
        
        log.debug('\n')


    df_fold = pd.DataFrame(results)
    models = df_fold['model_name'].unique()
    log.info('\n')
    log.info('-' * 30)
    for model in models:

        df_model = df_fold[df_fold['model_name'] == model]
        mean_f1 = float(f'{np.mean(df_model["F1"]):.2f}')

        log.info(f'MODEL {model} with .....: {mean_f1}')

    #data_results.append(df_fold)
    #log.info('-' * 30)

    #return pd.concat(data_results)
    return df_fold
 
NUM_ITER = 50
def process_A():
    x, y = read_A_thrombosis_non_thrombosis_v5()
    
    y.replace(to_replace=["Non_thrombosis", "Thrombosis"], value=[0, 1], inplace=True)

    # run
    df = run_experiment(x, y, NUM_ITER)
    df.to_csv('results/results_v4_v5_0.20/A_thrombosis_non_thrombosis_v5.csv', index=False)
    
    # only show mean metrics
    models = df['model_name'].unique()
    for model in models:

        df_model = df[df['model_name'] == model]
        mean_f1 = float(f'{np.mean(df_model["F1"]):.2f}')
        log.info(f'MODEL {model} with .....: {mean_f1}')

def process_B():
    x, y = read_B_type_I_PE_vs_Type_II_v4()
    
    y.replace(to_replace=["Type_I", "Type_II"], value=[0, 1], inplace=True)

    # run
    df = run_experiment(x, y, NUM_ITER)
    df.to_csv('results/results_v4_v5_0.20/B_type_I_PE_vs_Type_II_v4.csv', index=False)
    
    # only show mean metrics
    models = df['model_name'].unique()
    for model in models:

        df_model = df[df['model_name'] == model]
        mean_f1 = float(f'{np.mean(df_model["F1"]):.2f}')
        log.info(f'MODEL {model} with .....: {mean_f1}')



def process_C():

    x, y = read_C_type_I_vs_Type_II_no_PE_v4()

    #Change encoder
    """
    encoder = LabelEncoder()        
    y_enc = encoder.fit_transform(y)
    """

    y.replace(to_replace=["Type_I", "Type_II"], value=[0, 1], inplace=True)

    # run
    df = run_experiment(x, y, NUM_ITER)
    df.to_csv('results/results_v4_v5_0.20/C_type_I_vs_Type_II_no_PE_v4.csv', index=False)
    
    # only show mean metrics
    models = df['model_name'].unique()
    for model in models:

        df_model = df[df['model_name'] == model]
        mean_f1 = float(f'{np.mean(df_model["F1"]):.2f}')
        log.info(f'MODEL {model} with .....: {mean_f1}')


def process_D():

    x, y = read_D_type_I_PE_vs_Type_II_v4()

    #Change encoder
    """
    encoder = LabelEncoder()        
    y_enc = encoder.fit_transform(y)
    """

    y.replace(to_replace=["Type_I", "Type_II"], value=[0, 1], inplace=True)

    df = run_experiment(x, y, NUM_ITER)
    df.to_csv('results/results_v4_v5_0.20/D_type_I_vs_type_II_v4.csv', index=False)

    models = df['model_name'].unique()
    for model in models:

        df_model = df[df['model_name'] == model]
        mean_f1 = float(f'{np.mean(df_model["F1"]):.2f}')
        log.info(f'MODEL {model} with .....: {mean_f1}')
            

if __name__ == '__main__':

    """
    log.info('Process A')
    process_A()
    log.info('Finished A')
    log.info('Process B')
    process_B()
    log.info('Finished B')
    """

    log.info('Process C')
    process_C()
    log.info('Finished C')
    log.info('Process D')
    process_D()
    log.info('Finished D')
    log.info('Done.')
    

