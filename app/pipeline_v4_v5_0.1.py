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
from sklearn.metrics import RocCurveDisplay

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
    
    p = np.ceil(size_minority * percentage).astype('int')
    train = []
    test = []
    for classe in X[at].unique():
        
        df_class = X[X[at] == classe]
        
        test.append(df_class.iloc[:p])
        train.append(df_class.iloc[p:])
        
    df_train = pd.concat(train)
    df_test = pd.concat(test)
    
    # surffle
    df_train = shuffle(df_train, random_state=rs)
    df_test = shuffle(df_test, random_state=rs)

    y_train = df_train[at]
    y_test = df_test[at]
        
    x_train = df_train.drop([at], axis=1)
    x_test = df_test.drop([at], axis=1)   
    
    return x_train, y_train, x_test, y_test

def roc_calc_viz(classifier, X_test, y_test):
    viz = RocCurveDisplay.from_estimator(
                classifier,
                X_test,
                y_test,
                alpha=0.3,
                lw=1
            )
    
    return viz.fpr, viz.tpr, viz.roc_auc

def roc_calc_viz_pred(y_true, y_pred):
    viz = RocCurveDisplay.from_predictions(
                            y_true,
                            y_pred,
                            #alpha=0.3,
                            #lw=1
                        )

    return viz.fpr, viz.tpr, viz.roc_auc

def run_experiment(x, y, iterations) -> pd.DataFrame:

    #data_results = []
 

    results = {'model_name': [], 'iteration':[], 'F1':[], 
                'ROC':[],'acc-class-1':[],'acc-class-2':[], 'SEN':[], 
                'SPE':[], 'MCC':[], 'TPR': [], 'FPR':[], 'AUC': []}

    for i in tqdm(range(iterations)):

        #folds = create_folds(x, y, 2)

        #for j, fold in enumerate(folds):
            
        #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

        x_train, y_train, x_test, y_test = create_balanced_folds(x, y, rs=i, percentage=0.1)
        print(len(y_test))
        print(len(y_train))

        x_train = x_train.to_numpy()
        x_test = x_test.to_numpy()
        y_train = y_train.to_numpy()
        y_test = y_test.to_numpy()

        log.debug('-' * 30)
        log.debug(f'Iteration {i}')
        log.debug('-' * 30)
        
        
        y_pred, y_pred_prob, model = RSRF(x_train, y_train, x_test, y_test)
        sen, spe, f1, roc, _, _, mcc, _, _, _ = computer_scores(y_test, y_pred, y_pred_prob)
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
        log.debug(f'RF .......: {f1}')

        
        y_pred, y_pred_prob, model = RSDT(x_train, y_train, x_test, y_test)
        sen, spe, f1, roc, jac, fmi, mcc, fpr, tpr, thresh = computer_scores(y_test, y_pred, y_pred_prob)
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
        log.debug(f'DT .......: {f1}')

        y_pred, y_pred_prob, model = RSNN(x_train, y_train, x_test, y_test)
        sen, spe, f1, roc, jac, fmi, mcc, fpr, tpr, thresh = computer_scores(y_test, y_pred, y_pred_prob)
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
        log.debug(f'OneClassSVM ......: {f1}')
     
        y_pred, y_pred_prob, model = RSLocalOutlierFactor(x_train, y_train, x_test, y_test)
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
        log.debug(f'LocalOutlierFactor ......: {f1}')

        model, y_pred_prob, y_pred = RSIsolationForest(x_train, y_train, x_test, y_test)
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
        log.debug(f'IsolationForest ......: {f1}')

        y_pred, y_pred_prob, model = RSXgboost(x_train, y_train, x_test, y_test)
        sen, spe, f1, roc, jac, fmi, mcc, fpr, tpr, thresh = computer_scores(y_test, y_pred, y_pred_prob)
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
 
NUM_ITER = 3
def process_A():
    x, y = read_A_thrombosis_non_thrombosis_v5()
    
    y.replace(to_replace=["Non_thrombosis", "Thrombosis"], value=[0, 1], inplace=True)

    # run
    df = run_experiment(x, y, NUM_ITER)
    df.to_csv('results/results_v4_v5_0.1/A_thrombosis_non_thrombosis_v5_roc.csv', index=False)
    
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
    df.to_csv('results/results_v4_v5_0.1/B_type_I_PE_vs_Type_II_v4_roc.csv', index=False)
    
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
    df.to_csv('results/results_v4_v5_0.1/C_type_I_vs_Type_II_no_PE_v4_roc.csv', index=False)
    
    # only show mean metrics
    models = df['model_name'].unique()
    for model in models:

        df_model = df[df['model_name'] == model]
        mean_f1 = float(f'{np.mean(df_model["F1"]):.2f}')
        log.info(f'MODEL {model} with .....: {mean_f1}')


def process_D(test_percentage):

    x, y = read_D_type_I_PE_vs_Type_II_v4()

    y.replace(to_replace=["Type_I", "Type_II"], value=[0, 1], inplace=True)

    df = run_experiment(x, y, NUM_ITER)

    df.to_csv(f'results/results_v4_v5_marcos/D_type_I_vs_type_II_v4_roc_{test_percentage}.csv', index=False)

    response = {'MN': [], 'ACC_1': [], 'ACC_2': [], 'F1': [], 'AUC': [], 'MCC': []}
    
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
        response['MCC'].append(np.mean(df_model["MCC"]))

    df_describe = pd.DataFrame(response)
    log.info('-' * 30)
    log.info('DESCRIBE')
    log.info(df_describe)
    log.info('-' * 30)
    df_describe.to_csv(f'results/results_v4_v5_marcos/D_type_I_vs_type_II_v4_roc_{test_percentage}_describe.csv', index=False)
            

if __name__ == '__main__':

    """
    log.info('Process A')
    process_A()
    log.info('Finished A')
    log.info('Process B')
    process_B()
    log.info('Finished B')
    

    log.info('Process C')
    process_C()
    log.info('Finished C')
    """

    log.info('Process D')
    process_D()
    log.info('Finished D')
    log.info('Done.')
    

