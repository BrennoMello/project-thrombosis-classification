import sys
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
from synclass.sfc import SynClass

from ialovecoffe.data import *
from ialovecoffe.models import *
from ialovecoffe.validation import computer_scores, computer_scores_outlier, accuracy_per_class
from sklearn.utils import shuffle
from sklearn.metrics import RocCurveDisplay
from pipeline import undersampling
from pipeline import test_balacing

# config
warnings.simplefilter("ignore")
random.seed(10)

def double_maxclass_balance(X, Y, percentage, rs, at='target', sm=False):
    
    X[at] = Y

    # surffle
    X = shuffle(X, random_state=rs)
   
    size_minority = min(Counter(X[at]).values()) 
    
    p = np.ceil(size_minority * percentage).astype('int')
    p_train = (size_minority - p)
    qtd_max_class = (size_minority * 2) + p

    train = []
    test = []
    for classe in X[at].unique():
        
        df_class = X[X[at] == classe]
        if len(df_class) == max(Counter(X[at]).values()): 
            test.append(df_class.iloc[:p])
            train.append(df_class.iloc[p:qtd_max_class])
        else:
            test.append(df_class.iloc[:p])
            train.append(df_class.iloc[p:p_train])
        
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

    #data_results = []

    results = {'model_name': [], 'iteration':[], 'F1':[], 
                'ROC':[],'acc-class-1':[],'acc-class-2':[], 'SEN':[], 
                'SPE':[], 'MCC':[], 'TPR': [], 'FPR':[], 'AUC': [], 'ATT': []}

    for i in tqdm(range(iterations)):

        x_train_raw, y_train_raw, x_test_raw, y_test_raw = test_balacing(x, y, p, i, False)

        #fitting a model per attribute
        for att in np.arange(x_train_raw.shape[1]):

            x_train = x_train_raw.iloc[:, att].values.reshape(-1, 1)
            x_test = x_test_raw.iloc[:, att].values.reshape(-1, 1)
            y_train = y_train_raw.to_numpy()
            y_test = y_test_raw.to_numpy()

            log.debug('-' * 20)
            log.debug(f'Att {att} - {dsetname} - Iteration {i} - test size: {p}')
            log.debug('-' * 20)

            y_pred, y_pred_prob, model = RSRF(x_train, y_train, x_test, y_test)
            sen, spe, f1, roc, jac, fmi, mcc= computer_scores(y_test, y_pred)
            acc1, acc2 = accuracy_per_class(y_test, y_pred)
            viz_fpr, viz_tpr, viz_auc = roc_calc_viz_pred(y_test, y_pred_prob[:,1])
            results['model_name'].append('RandomForest')
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
            results['ATT'].append(att)
            log.debug(f'RF .......: {f1}')
            
            y_pred, y_pred_prob, model = RSDT(x_train, y_train, x_test, y_test)
            sen, spe, f1, roc, jac, fmi, mcc = computer_scores(y_test, y_pred)
            acc1, acc2 = accuracy_per_class(y_test, y_pred)
            viz_fpr, viz_tpr, viz_auc = roc_calc_viz_pred(y_test, y_pred_prob[:,1])
            results['model_name'].append('DecisionTree')
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
            results['ATT'].append(att)
            log.debug(f'DT .......: {f1}')

            y_pred, y_pred_prob, model = RSNN(x_train, y_train, x_test, y_test)
            sen, spe, f1, roc, jac, fmi, mcc = computer_scores(y_test, y_pred)
            acc1, acc2 = accuracy_per_class(y_test, y_pred)
            viz_fpr, viz_tpr, viz_auc = roc_calc_viz_pred(y_test, y_pred_prob[:,1])
            results['model_name'].append('KNN')
            results['iteration'].append(i)
            results['F1'].append(f1)
            results['acc-class-1'].append(acc1)
            results['acc-class-2'].append(acc2)
            results['ROC'].append(roc)
            results['SEN'].append(sen)
            results['SPE'].append(spe)
            results['MCC'].append(mcc)
            results['FPR'].append(viz_fpr)
            results['TPR'].append(viz_tpr)
            results['AUC'].append(viz_auc)
            results['ATT'].append(att)
            log.debug(f'KNN ......: {f1}')
            
            y_pred, y_pred_prob, model = RSOneClassSVM(x_train, y_train, x_test, y_test)
            sen, spe, f1, roc, jac, fmi, mcc, fpr, tpr, thresh = computer_scores_outlier(y_test, y_pred)
            acc1, acc2 = accuracy_per_class(y_test, y_pred)
            viz_fpr, viz_tpr, viz_auc = roc_calc_viz_pred(y_test, y_pred_prob)
            results['model_name'].append('OneClassSVM')
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
            results['ATT'].append(att)
            log.debug(f'OneClassSVM ......: {f1}')
        
            y_pred, y_pred_prob, model = RSLocalOutlierFactor_ml(x_train, y_train, x_test, y_test)
            sen, spe, f1, roc, jac, fmi, mcc, fpr, tpr, thresh = computer_scores_outlier(y_test, y_pred)
            acc1, acc2 = accuracy_per_class(y_test, y_pred)
            viz_fpr, viz_tpr, viz_auc = roc_calc_viz_pred(y_test, y_pred_prob)
            results['model_name'].append('LocalOutlierFactor')
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
            results['ATT'].append(att)
            log.debug(f'LocalOutlierFactor ......: {f1}')

            y_pred, y_pred_prob, model= RSIsolationForest(x_train, y_train, x_test, y_test)
            sen, spe, f1, roc, jac, fmi, mcc, fpr, tpr, thresh = computer_scores_outlier(y_test, y_pred)
            acc1, acc2 = accuracy_per_class(y_test, y_pred)
            viz_fpr, viz_tpr, viz_auc = roc_calc_viz_pred(y_test, y_pred_prob)
            results['model_name'].append('IsolationForest')
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
            results['ATT'].append(att)
            log.debug(f'IsolationForest ......: {f1}')

            y_pred, y_pred_prob, model = RSXgboost(x_train, y_train, x_test, y_test)
            sen, spe, f1, roc, jac, fmi, mcc = computer_scores(y_test, y_pred)
            acc1, acc2 = accuracy_per_class(y_test, y_pred)
            viz_fpr, viz_tpr, viz_auc = roc_calc_viz_pred(y_test, y_pred_prob[:,1])
            results['model_name'].append('Xgboost')
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
            results['ATT'].append(att)
            log.debug(f'Xgboost ......: {f1}')

            """
            y_pred, y_pred_prob, model = SVM_hiperopt(x_train, y_train, x_test, y_test)
            sen, spe, f1, roc, jac, fmi, mcc = computer_scores(y_test, y_pred)
            acc1, acc2 = accuracy_per_class(y_test, y_pred)
            viz_fpr, viz_tpr, viz_auc = roc_calc_viz_pred(y_test, y_pred_prob[:,1])
            results['model_name'].append('SVM')
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
            results['ATT'].append(att)
            log.debug(f'SVM ......: {f1}')
            """

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
 






