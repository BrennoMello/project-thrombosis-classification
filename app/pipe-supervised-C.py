# io, system, data
import sys
import time
import pickle
import warnings
import random
import pandas as pd
import numpy as np
from loguru import logger as log
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
from datetime import datetime

# preprocessing
from ialovecoffe.data import pre_processing, data_splitting
from ialovecoffe.data import read_dataset_C

# models
from ialovecoffe.models import *

# metrics - validation
from ialovecoffe.validation import computer_scores, accuracy_per_class

simplefilter("ignore", category=ConvergenceWarning)


def run_pipe():

    random.seed(10)
    n_iter = 10
    NF = 10

    results = {'model_name': [], 'iteration':[], 'fold':[], 'F1':[], 
               'ROC':[],'acc-class-1':[],'acc-class-2':[], 'SEN':[], 'SPE':[]}

    norm, stand, smote = True, False, True

    df = read_dataset_C()

    df['type'] = df['type'].replace(['Type_I', 'Type_II'], [1, -1]).to_numpy()
    
    # pre processed data
    x, y, df = pre_processing(df, norm=norm, stand=stand)

    ini = time.time()
    
    # iterations
    for i in np.arange(n_iter):

        datafds = data_splitting(x, y, df, n_folds=NF,  smote=smote, test_size=5)
        NF = len(datafds)
        # cross validation
        for j in range(NF): 
            # get data
            x_train, y_train, x_test, y_test = datafds[j]
            # save data
            sav = True #save_data_fold(i, j, x_train, y_train, x_test, y_test)
            
            log.info(f'Iter: {i} save fold {j} saved:{sav}')
            
            model, y_pred = RSRF(x_train, y_train, x_test, y_test)
            sen, spe, f1, roc, jac, fmi, mcc = computer_scores(y_test, y_pred)
            acc1, acc2 = accuracy_per_class(y_test, y_pred)
            results['model_name'].append('RandomForestClassifier')
            results['iteration'].append(i)
            results['fold'].append(j)
            results['acc-class-1'].append(acc1)
            results['acc-class-2'].append(acc2)
            results['F1'].append(f1)
            results['ROC'].append(roc)
            results['SEN'].append(sen)
            results['SPE'].append(spe)
            log.info(f'RandomForestClassifier..: {f1} acc class 1: {acc1} acc class 2: {acc2}')
            save_model(model, filename=f"RF_iter-{i}_fold-{j}_f1-{f1}.pkl", 
                              folder='saved_models/C/')
            
            
            model, y_pred = RSNN(x_train, y_train, x_test, y_test)
            sen, spe, f1, roc, jac, fmi, mcc = computer_scores(y_test, y_pred)
            acc1, acc2 = accuracy_per_class(y_test, y_pred)
            results['model_name'].append('KNN')
            results['iteration'].append(i)
            results['fold'].append(j)
            results['F1'].append(f1)
            results['acc-class-1'].append(acc1)
            results['acc-class-2'].append(acc2)
            results['ROC'].append(roc)
            results['SEN'].append(sen)
            results['SPE'].append(spe)
            log.info(f'KNN ....................: {f1}')
            save_model(model, filename=f"KNN_iter-{i}_fold-{j}_f1-{f1}.pkl", 
                              folder='saved_models/C/')
            
            
            model, y_pred = RSDT(x_train, y_train, x_test, y_test)
            sen, spe, f1, roc, jac, fmi, mcc = computer_scores(y_test, y_pred)
            acc1, acc2 = accuracy_per_class(y_test, y_pred)
            results['model_name'].append('DecisionTree')
            results['iteration'].append(i)
            results['fold'].append(j)
            results['acc-class-1'].append(acc1)
            results['acc-class-2'].append(acc2)
            results['F1'].append(f1)
            results['ROC'].append(roc)
            results['SEN'].append(sen)
            results['SPE'].append(spe)
            log.info(f'DT .....................: {f1}')
            save_model(model, filename=f"DT_iter-{i}_fold-{j}_f1-{f1}.pkl", 
                              folder='saved_models/C/')

            model, y_pred = RSSVM(x_train, y_train, x_test, y_test)
            sen, spe, f1, roc, jac, fmi, mcc = computer_scores(y_test, y_pred)
            acc1, acc2 = accuracy_per_class(y_test, y_pred)
            results['model_name'].append('SVM')
            results['iteration'].append(i)
            results['fold'].append(j)
            results['acc-class-1'].append(acc1)
            results['acc-class-2'].append(acc2)
            results['F1'].append(f1)
            results['ROC'].append(roc)
            results['SEN'].append(sen)
            results['SPE'].append(spe)
            log.info(f'SVM ....................: {f1}')
            save_model(model, filename=f"SVM_iter-{i}_fold-{j}_f1-{f1}.pkl", 
                              folder='saved_models/C/')
            

    log.info(f'process in :{(time.time()-ini)/60:.3f} minutes')

            
    dt_string = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    # at the end, save results
    df = pd.DataFrame(results)

    # show sample
    log.info('-' * 30)
    log.info('-' * 30)
    log.info(df.describe())
    log.info('-' * 30)


    folder = 'results/'

    df.to_csv(f'{folder}C-RS-ALL-{n_iter}_Fds-{NF}_norm-{norm}_stand-{stand}_smote-{smote}_{dt_string}.csv', index=False)


if __name__ == '__main__':
    sys.path.append('')
    run_pipe()
