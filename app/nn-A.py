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
from ialovecoffe.data import read_dataset_A, save_data_fold
# models
from ialovecoffe.models import *
from tensorflow import keras
# metrics - validation
from ialovecoffe.validation import computer_scores as cs
from ialovecoffe.validation import accuracy_per_class
from sklearn.metrics import confusion_matrix

simplefilter("ignore", category=ConvergenceWarning)


def get_scores(scores, model):
    
    metrics = model.metrics_names
    precision = scores[5]
    recall = scores[-1]
    F1 = 2 * (precision * recall) / (precision + recall)
    
    for i, metric in enumerate(metrics):
        log.info(f'{metric}:{scores[i]:.2f}')

    return F1

def create_model(shape_in, shape_out=1):
    model = keras.Sequential(
    [
        keras.layers.Dense(
            256, activation="relu", input_shape=(shape_in.shape[-1],)
        ),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(shape_out, activation="sigmoid"),
        ]
    )

    metrics = [
        keras.metrics.FalseNegatives(name="fn"),
        keras.metrics.FalsePositives(name="fp"),
        keras.metrics.TrueNegatives(name="tn"),
        keras.metrics.TruePositives(name="tp"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        ]

    model.compile(
        optimizer=keras.optimizers.Adam(1e-2), loss="binary_crossentropy", metrics=metrics
    )
    
    return model


def run_experiment():
    
    df = read_dataset_A()
    df['type'] = df['type'].replace(['Non_thrombosis', 'Thrombosis'], [0, 1]).to_numpy()
    df.head()

    train_targets = df['type']

    counts = np.bincount(train_targets)
    per = 100 * float(counts[1]) / len(train_targets)
    print(f"Number of positive samples in training data: {counts[1]} ({per:.2f}% of total)")

    weight_for_0 = 1.0 / counts[0]
    weight_for_1 = 1.0 / counts[1]
    log.info(f' w c1: {weight_for_0} w c2: {weight_for_1}')

    random.seed(10)
    n_iter = 10
    NF = 'auto'

    #results = {'model_name': [], 'iteration':[], 'fold':[], 'F1':[], 
    #            'ROC':[],'acc-class-1':[],'acc-class-2':[], 'SEN':[], 'SPE':[]}
    results = {'iteration':[], 'fold': [], 'f1':[]}

    norm, stand, smote = True, False, True

    x, y, df = pre_processing(df, norm=norm, stand=stand)

    for i in range(n_iter):

        datafds = data_splitting(x, y, df, test_size=14,  smote=smote)

        NF = len(datafds)

        for j in range(NF): 
            # get data
            x_train, y_train, x_test, y_test = datafds[j]

            nn = create_model(x_train)

            callbacks = [keras.callbacks.ModelCheckpoint("weigths/thombose_model_at_epoch_{epoch}.h5")]
            class_weight = {0: weight_for_0, 1: weight_for_1}

            nn.fit(
                x_train,
                y_train,
                batch_size=32,
                epochs=30,
                verbose=0,
                callbacks=callbacks,
                validation_split=0.1,
                class_weight=class_weight,
            )

            y_pred = nn.predict(x_test)
            y_pred = np.where(y_pred <= 0.5, 0, 1)[:,0]
            scores = nn.evaluate(x_test, y_test)

            f1 = get_scores(scores, nn)

            # print other scores:
            sen, spe, f1, roc, jac, fmi, mcc = cs(y_test, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            log.info('-' * 30)
            log.info(f'Scores by sklearn')
            log.info(f'fn:{fn}')
            log.info(f'fp:{fp}')
            log.info(f'tn:{tn}')
            log.info(f'tp:{tp}')
            log.info('-' * 30)
            log.info(f'Iter: {i} fold: {j} f1 keras: {f1} f1 sklearn: {f1}')
            log.info("=" * 30)

            results['iteration'].append(i)
            results['fold'].append(j)
            results['f1'].append(f1)



    
    log.info(f'Mean F1:{np.mean(results["f1"])}')
    print('-'*30)
    log.info(f'Results')
    log.info(f'=> {results}')
            
run_experiment()