import os
import sys
#print(os.getcwd())
#
#print(os.listdir())
#sys.path.append('../../')


sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
from loguru import logger as log
#from pipeline import run_experiment
from data import read_dataset_v
from tqdm import tqdm
from collections import Counter

from ialovecoffe.data import *
from ialovecoffe.models import *
from ialovecoffe.validation import computer_scores, computer_scores_outlier, accuracy_per_class
from sklearn.utils import shuffle
from sklearn.metrics import RocCurveDisplay


def undersampling(X, Y, percentage, rs, at='target', increase=1, sm=False):
    
    # set target in dataframe
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




def run_experiment_stack(x, y, iterations, p, dsetname) -> pd.DataFrame:

    #data_results = []

    results = {'model_name': [], 'iteration':[], 'F1':[], 
                'ROC':[],'acc-class-1':[],'acc-class-2':[], 'SEN':[], 
                'SPE':[], 'MCC':[], 'TPR': [], 'FPR':[], 'AUC': []}

    for i in tqdm(range(iterations)):

        x_train_raw, y_train_raw, x_test_raw, y_test_raw = undersampling(x, y, p, i, False)

        x_train = x_train_raw.to_numpy()
        x_test = x_test_raw.to_numpy()
        y_train = y_train_raw.to_numpy()
        y_test = y_test_raw.to_numpy()

        log.debug('-' * 30)
        log.debug(f'{dsetname} - Iteration {i} - test size: {p}')
        log.debug('-' * 30)


        y_pred, y_pred_prob, model = stacking_five_models(x_train, y_train, x_test, y_test)
        sen, spe, f1, roc, jac, fmi, mcc = computer_scores(y_test, y_pred)
        acc1, acc2 = accuracy_per_class(y_test, y_pred)
        viz_fpr, viz_tpr, viz_auc = roc_calc_viz_pred(y_test, y_pred_prob[:,1])
        results['model_name'].append('Stacking')
        results['iteration'].append(i)
        results['acc-class-1'].append(acc1)
        results['acc-class-2'].append(acc2)
        results['F1'].append(f1)
        results['ROC'].append(roc)
        results['SEN'].append(sen)
        results['SPE'].append(spe)
        results['MCC'].append(mcc)
        results['FPR'].append(viz_fpr)
        results['TPR'].append(viz_tpr)
        results['AUC'].append(viz_auc)
        log.debug(f'Stacking ......: {f1}')

        log.debug('\n')


    df_fold = pd.DataFrame(results)
    models = df_fold['model_name'].unique()
    log.info('\n')
    log.info('-' * 30)
    for model in models:

        df_model = df_fold[df_fold['model_name'] == model]
        mean_f1 = float(f'{np.mean(df_model["F1"]):.2f}')

        log.info(f'MODEL {model} with .....: {mean_f1}')

    return df_fold


def process_v(test_percentage, NUM_ITER = 10, folder='fator_v'):

    x, y = read_dataset_v()

    y = np.where(y > 0, 1, 0)
    #y.replace(to_replace=["no", "yes"], value=[0, 1], inplace=True)
    # run
    df = run_experiment_stack(x, y, NUM_ITER, test_percentage, 'factor_v')
    df.to_csv(f'results/{folder}/src_bal_FV_final_dataset_clean_{test_percentage}.csv', index=False)
    
    response = {'MN': [], 'ACC_1': [], 'ACC_2': [], 'F1': [], 'AUC': []}
    
    # only show mean metrics
    models = df['model_name'].unique()
    for model in models:

        df_model = df[df['model_name'] == model]
        mean_f1 = float(f'{np.mean(df_model["F1"]):.2f}')
        log.info(f'MODEL {model} with .....: {mean_f1}')

        response['MN'].append(model)
        response['ACC_1'].append(np.mean(df_model["acc-class-1"]))
        response['ACC_2'].append(np.mean(df_model["acc-class-2"]))
        response['F1'].append(np.mean(df_model["F1"]))
        response['AUC'].append(np.mean(df_model["ROC"]))
    
    # ---------------------------------
    # ---------------------------------
    # ---------------------------------
    df_describe = pd.DataFrame(response)
    log.info('-' * 30)
    log.info('DESCRIBE')
    log.info(df_describe)
    log.info('-' * 30)
    df_describe.to_csv(f'results/{folder}/src_bal_ml_FV_final_dataset_clean_{test_percentage}_describe.csv', index=False)


if __name__ == '__main__':
    ITERS = 50
    log.add("experiment_src_ml_undersample_same_class.log")
    log.info('-' * 30)
    log.info('THROMBOSE Detection with Machine Learning 1 times')
    log.info('-' * 30)

    for test_size in [0.1, 0.15, 0.2, 0.25]:
        process_v(test_size, NUM_ITER=ITERS, folder='stacking')
       
