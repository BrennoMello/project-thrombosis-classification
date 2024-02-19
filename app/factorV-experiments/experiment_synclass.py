import numpy as np
import pandas as pd
from loguru import logger as log
from pipeline_synclass import run_experiment
from data import read_dataset_v
from tqdm import tqdm
from collections import Counter
            
def process_v(test_percentage, NUM_ITER = 5, folder='results'):

    x, y = read_dataset_v()

    y = np.where(y > 0, 1, 0)
    #y.replace(to_replace=["no", "yes"], value=[0, 1], inplace=True)
    
    # run
    df = run_experiment(x, y, NUM_ITER, test_percentage, 'synclass')
    df.to_csv(f'results/{folder}/balanced_FV_final_dataset_clean_{test_percentage}.csv', index=False)
    
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
    df_describe.to_csv(f'results/{folder}/balanced_FV_final_dataset_clean_{test_percentage}_describe.csv', index=False)

if __name__ == '__main__':
    ITERS = 100
    log.add("experiment_syn_class_lof_3_times.log")
    log.info('-' * 30)
    log.info('Factor V Detection with Machine Learning with synclass')
    log.info('-' * 30)

    """
    for test_size in [0.1, 0.15, 0.2, 0.25]:
        
        process_A(test_size, NUM_ITER=ITERS, folder='results-synclass-dt')
        process_B(test_size, NUM_ITER=ITERS, folder='results-synclass-dt')
        process_C(test_size, NUM_ITER=ITERS, folder='results-synclass-dt')
        process_D(test_size, NUM_ITER=ITERS, folder='results-synclass-dt')
        
        process_v(test_size, NUM_ITER=ITERS, folder='synclass')
    """

    process_v(test_percentage=0.15, NUM_ITER=ITERS, folder='synclass/model_search')