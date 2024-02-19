import os
import random
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from loguru import logger as log

#
# NEW DATASETS
#

def read_A_thrombosis_non_thrombosis_v5(path_src = 'ialovecoffe/inputSets_for_thrombosis_pipeline/'):
    df = pd.read_csv(f'{path_src}thrombosis_non_thrombosis_v5.csv', sep='\t')

    Y = df['type'].copy()
    df.drop(['node', 'type'], inplace=True, axis=1)
    X = df
    return X, Y

def read_B_type_I_PE_vs_Type_II_v4(path_src = 'ialovecoffe/inputSets_for_thrombosis_pipeline/'):
    df = pd.read_csv(f'{path_src}type_I_PE_vs_Type_II_v4.csv', sep='\t')

    Y = df['type'].copy()
    df.drop(['node', 'type'], inplace=True, axis=1)
    X = df
    return X, Y

def read_C_type_I_vs_Type_II_no_PE_v4(path_src = 'ialovecoffe/inputSets_for_thrombosis_pipeline/'):
    df = pd.read_csv(f'{path_src}type_I_vs_Type_II_no_PE_v4.csv', sep='\t')
    
    Y = df['type'].copy()
    df.drop(['type', 'node'], inplace=True, axis=1)
    X = df
    return X, Y


def read_D_type_I_PE_vs_Type_II_v4(path_src = 'ialovecoffe/inputSets_for_thrombosis_pipeline/'):
    df = pd.read_csv(f'{path_src}type_I_PE_vs_Type_II_v4.csv', sep='\t')
    
    Y = df['type'].copy()
    df.drop(['type', 'node'], inplace=True, axis=1)
    X = df
    return X, Y

# ----------------------------------------------------------

def read_A_thrombosis_non_thrombosis_v4(path_src = 'ialovecoffe/final_data/'):
    df = pd.read_csv(f'{path_src}thrombosis_non_thrombosis_v4.csv', sep='\t')

    Y = df['type'].copy()
    df.drop(['node', 'type'], inplace=True, axis=1)
    X = df
    return X, Y

def read_B_type_I_PE_vs_Type_II_v3(path_src = 'ialovecoffe/final_data/'):
    df = pd.read_csv(f'{path_src}type_I_PE_vs_Type_II_v3.csv', sep='\t')

    Y = df['type'].copy()
    df.drop(['node', 'type'], inplace=True, axis=1)
    X = df
    return X, Y

def read_C_type_I_vs_Type_II_no_PE_v3(path_src = 'ialovecoffe/final_data/'):
    df = pd.read_csv(f'{path_src}type_I_vs_Type_II_no_PE_v3.csv', sep='\t')
    
    Y = df['type'].copy()
    df.drop(['type', 'node'], inplace=True, axis=1)
    X = df
    return X, Y


def read_D_type_I_PE_vs_Type_II_v3(path_src = 'ialovecoffe/final_data/'):
    df = pd.read_csv(f'{path_src}type_I_PE_vs_Type_II_v3.csv', sep='\t')
    
    Y = df['type'].copy()
    df.drop(['type', 'node'], inplace=True, axis=1)
    X = df
    return X, Y


# -----------------------------


def read_dataset_A():
    dataset = pd.read_csv("data/A-thrombosis_non_thrombosis.csv", delim_whitespace=True, header=0)
    
    return dataset.drop(['node'], axis=1)


def read_dataset_B():
    dataset = pd.read_csv("data/B-type_I_type_II.csv", delim_whitespace=True, header=0)
    
    return  dataset.drop(['node'], axis=1)


def read_dataset_C():
    dataset = pd.read_csv("data/C-type_I_PE_vs_Type_II.csv", delim_whitespace=True, header=0)
    
    return dataset.drop(['node'], axis=1)


def read_dataset_D():
    dataset = pd.read_csv("data/D-type_I_Type_II_no_PE.csv", delim_whitespace=True, header=0)
    
    return dataset.drop(['node'], axis=1)



def apply_smote(x, y):
    x_smt, y_smt = SMOTE().fit_resample(x, y)
    
    return  x_smt, y_smt


def create_cv_balanced(data_labels, test_size = 14, reference=1, folds = None):    
    
    index_thrombo = (np.where(data_labels == reference)[0]).tolist()
    index_no_thrombo = (np.where(data_labels != reference)[0]).tolist()  
    
    cv_indexes_thrombo = []
    cv_indexes_no_thrombo = []

    if folds == None:
        folds = int((min([len(index_thrombo), len(index_no_thrombo)]) / test_size))

    log.debug(f'creating {folds} folds.')

    for i in np.arange(folds-1):

        temp = random.sample(index_thrombo, test_size)
        temp_out = random.sample(index_no_thrombo, test_size)
        
        cv_indexes_thrombo.append(temp)
        cv_indexes_no_thrombo.append(temp_out)
        
        index_thrombo = [x for x in index_thrombo if x not in temp]
        index_no_thrombo = [x for x in index_no_thrombo if x not in temp_out]

    cv_indexes_thrombo.append(index_thrombo)
    cv_indexes_no_thrombo.append(index_no_thrombo)

    return cv_indexes_thrombo, cv_indexes_no_thrombo


def data_splitting(x, y, data_frame, 
                   n_folds = None, 
                   smote=False, 
                   test_size=14):

    data_folds = []
    
    # obter os 10
    index_yes, index_no = create_cv_balanced(data_frame['type'], 
                test_size=test_size, folds = n_folds)
    
    if n_folds == None:
        n_folds = len(index_yes)
    
    # formatar os indices em treino e teste
    for fold_id in range(n_folds):

        list_index = np.concatenate((index_yes[fold_id], index_no[fold_id])) #  .astype('int')

        # test
        x_test = x[list_index,].copy()
        y_test = y[list_index, ].copy()

        
        # train
        x_train = np.delete(x, list_index, axis=0)
        y_train = np.delete(y, list_index,)
            
        if smote:
            x_train, y_train = apply_smote(x_train, y_train)

        data_folds.append([x_train, y_train, x_test, y_test])

    return data_folds


def pre_processing(df, norm=True, 
                       stand=False, 
                       algorithm=None, 
                       k_best=21, 
                       atts=None):

    
    data_train_y = df['type'].to_numpy()
    
    if atts:
        data_train_X = df[atts].to_numpy() # set attributes
        data_train_X = data_train_X.reshape(data_train_X.shape[0], len(atts))
    else: 
        data_train_X = df.drop(['type'], axis=1).to_numpy()
    
    if norm:
        data_train_X = MinMaxScaler().fit_transform(data_train_X)

    if stand:
        data_train_X = StandardScaler().fit_transform(data_train_X)

    if algorithm:
        data_train_X = feature_selection(algorithm, k_best, data_train_X, data_train_y)

    return data_train_X, data_train_y, df


def save_data_fold(iteration, fold, x_train, y_train, x_test, y_test, folder='folds/'):
    try:
        folder_name = f'{folder}iter-{iteration}-fold-{fold}/'
            
        # create folder with metaname
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)

        # salve x and y
        np.save(folder_name+'x_train.npy', x_train)
        np.save(folder_name+'y_train.npy', y_train)
        np.save(folder_name+'x_test.npy', x_test)
        np.save(folder_name+'y_test.npy', y_test)

        return True
    except:
        return False