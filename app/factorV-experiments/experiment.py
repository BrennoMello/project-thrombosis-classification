import sys
sys.path.append('../')
import numpy as np
import pandas as pd
from ialovecoffe.data import *
from loguru import logger as log
from pipeline import run_experiment
from data import read_dataset_v
from tqdm import tqdm
from collections import Counter


def process_v(test_percentage, NUM_ITER = 10, folder='fator_v'):

    x, y = read_dataset_v()

    y = np.where(y > 0, 1, 0)
    #y.replace(to_replace=["no", "yes"], value=[0, 1], inplace=True)
    print(Counter(y))
    # run
    df = run_experiment(x, y, NUM_ITER, test_percentage, 'factor_v')
    df.to_csv(f'results/{folder}/random_FV_final_dataset_clean_{test_percentage}.csv', index=False)
    
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
        

    df_describe = pd.DataFrame(response)
    log.info('-' * 30)
    log.info('DESCRIBE')
    log.info(df_describe)
    log.info('-' * 30)
    df_describe.to_csv(f'results/{folder}/random_FV_final_dataset_clean_{test_percentage}_describe.csv', index=False)


if __name__ == '__main__':
    ITERS = 100
    log.add("experiment_3_times.log")
    log.info('-' * 30)
    log.info('Factor V Detection with Machine Learning 3 times')
    log.info('-' * 30)

    for test_size in [0.1, 0.15, 0.2, 0.25]:
        process_v(test_size, NUM_ITER=ITERS, folder='ml/undersample')
       