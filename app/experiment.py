import numpy as np
import pandas as pd
from ialovecoffe.data import *
from loguru import logger as log
from pipeline import run_experiment
from tqdm import tqdm

def process_A(test_percentage, NUM_ITER = 5, folder='all-results'):

    x, y = read_A_thrombosis_non_thrombosis_v5()
    
    y.replace(to_replace=["Non_thrombosis", "Thrombosis"], value=[0, 1], inplace=True)

    # run
    df = run_experiment(x, y, NUM_ITER, test_percentage, 'A')
    df.to_csv(f'results/{folder}/A_thrombosis_non_thrombosis_v5_{test_percentage}.csv', index=False)
    
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
    df_describe.to_csv(f'results/{folder}/A_thrombosis_non_thrombosis_v5_{test_percentage}_describe.csv', index=False)


def process_B(test_percentage, NUM_ITER = 5, folder='all-results'):

    x, y = read_B_type_I_PE_vs_Type_II_v4()
    
    y.replace(to_replace=["Type_I", "Type_II"], value=[0, 1], inplace=True)

    # run
    df = run_experiment(x, y, NUM_ITER, test_percentage, 'B')
    df.to_csv(f'results/{folder}/B_type_I_PE_vs_Type_II_v4_{test_percentage}.csv', index=False)
    
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
    df_describe.to_csv(f'results/{folder}/B_type_I_PE_vs_Type_II_v4_{test_percentage}_describe.csv', index=False)


def process_C(test_percentage, NUM_ITER = 5, folder='all-results'):
    
    x, y = read_C_type_I_vs_Type_II_no_PE_v4()

    y.replace(to_replace=["Type_I", "Type_II"], value=[0, 1], inplace=True)

    # run
    df = run_experiment(x, y, NUM_ITER, test_percentage, 'C')
    df.to_csv(f'results/{folder}/C_type_I_vs_Type_II_no_PE_v4_{test_percentage}.csv', index=False)
    
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
    df_describe.to_csv(f'results/{folder}/C_type_I_vs_Type_II_no_PE_v4_{test_percentage}_describe.csv', index=False)


def process_D(test_percentage, NUM_ITER = 5, folder='all-results'):

    x, y = read_D_type_I_PE_vs_Type_II_v4()

    y.replace(to_replace=["Type_I", "Type_II"], value=[0, 1], inplace=True)

    # run
    df = run_experiment(x, y, NUM_ITER, test_percentage, 'D')
    df.to_csv(f'results/{folder}/D_type_I_vs_type_II_v4_roc_{test_percentage}.csv', index=False)
    
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
    df_describe.to_csv(f'results/{folder}/D_type_I_vs_type_II_v4_roc_{test_percentage}_describe.csv', index=False)
            

if __name__ == '__main__':
    ITERS = 50
    log.info('-' * 30)
    log.info('THROMBOSE Detection with Machine Learning')
    log.info('-' * 30)

    for test_size in [0.1, 0.15, 0.2, 0.25]:
        process_A(test_size, NUM_ITER=ITERS, folder='')
        process_B(test_size, NUM_ITER=ITERS, folder='')
        process_C(test_size, NUM_ITER=ITERS, folder='')
        process_D(test_size, NUM_ITER=ITERS, folder='')
