#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

from sfc import *

datasets = ['../../data/thrombosis_non_thrombosis_v4.csv',
           '../../data/type_I_vs_type_II_v3.csv',
           '../../data/type_I_PE_vs_Type_II_v3.csv',
           '../../data/type_I_vs_Type_II_no_PE_v3.csv']

acc_01 = []
acc_02 = []
f1_all = []
acc_01_sd = []
acc_02_sd = []
f1_all_sd = []
n_c1_mean = []
n_c2_mean = []
auc_all_mu = []
auc_all_std = []

dataset_names = []
threshold_values = []

testSize_smaller = 10
#rep = 1
rep = 50

for ds in datasets:
    for t in np.arange(start=0.4, stop=0.81, step=0.01):  
        print(f'Analyzing the dataset:{ds} using threshold {t}')
        pos_lab = 'Type_II'

        if "thrombosis" in ds:
            pos_lab = 'Thrombosis'


        biodata = pd.read_csv(ds, sep="\t")
        biodata.drop(['node'], axis='columns', inplace=True)
        
        # normalizing...
        for i in np.arange(biodata.shape[1]-1):
            biodata.iloc[:, i] = (biodata.iloc[:, i] - min(biodata.iloc[:, i]))/(max(biodata.iloc[:, i]) - min(biodata.iloc[:, i]))

        acc, f1, n_inst, auc = runSFC(biodata, nrepetitions = rep, 
                            testSize_smaller_class = testSize_smaller, positive=pos_lab, classThreshold = t)

        acc_01.append(np.round(np.mean(acc[:,0]), 2))
        acc_02.append(np.round(np.mean(acc[:,1]), 2))
        f1_all.append(np.round(np.mean(f1),2))

        acc_01_sd.append(np.round(np.std(acc[:,0]), 2))
        acc_02_sd.append(np.round(np.std(acc[:,1]), 2))
        f1_all_sd.append(np.round(np.std(f1),2))

        n_c1_mean.append(np.round(np.mean(n_inst[:,0]/testSize_smaller),2))
        n_c2_mean.append(np.round(np.mean(n_inst[:,1]/testSize_smaller),2))

        dataset_names.append(ds)
        threshold_values.append(t)
        auc_all_mu.append(np.round(np.mean(auc),2))
        auc_all_std.append(np.round(np.std(auc),2))

total_data = {'Threshold': threshold_values, 
            'Dataset': dataset_names, 
            'Mean accuracy - Class 01': acc_01,
            'Mean accuracy - Class 02': acc_02,
            'Mean F1': f1_all,
            'Std accuracy - Class 01': acc_01_sd,
            'Std accuracy - Class 02': acc_02_sd,
            'Std F1': f1_all_sd,
            'Mean AUC': auc_all_mu,
            'Std AUC': auc_all_std,
            'Mean C1 Instances Remaining': n_c1_mean,
            'Mean C2 Instances Remaining': n_c2_mean}

df = pd.DataFrame(data=total_data)
df.to_csv('/tmp/results_output.csv', mode='w')

