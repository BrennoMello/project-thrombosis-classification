import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import r
from rpy2.robjects import numpy2ri
import xgboost as xgb
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn import metrics
from collections import Counter
#import random
#from collections import Counter

def call_wilcox_R (x, y):
    r.assign("x", x.to_numpy())
    r.assign("y", y.to_numpy())
    r('res<-wilcox.test(x~y)$statistic')
    r_result = r("res")
    return (r_result[0])

def get_statistical_weights(inputSet, ignore_class = np.nan):
    
    myWeights = np.repeat(np.nan, inputSet.shape[1]-1)

    if(np.isnan(ignore_class)):
        ignore_class =  inputSet.shape[1]-1

    numpy2ri.activate()
    coln = np.delete(np.arange(inputSet.shape[1]), ignore_class)    
    for i in np.arange(len(coln)):
        myWeights[i] = call_wilcox_R(inputSet.iloc[:,coln[i]],
                            inputSet.iloc[:,ignore_class])

        
    numpy2ri.deactivate()

    return (1/myWeights)

def runXGBoost(train_fold, test_fold, positive, negative, metric = 'auc', class_idx = np.nan):
    if(np.isnan(class_idx)):
        class_idx =  train_fold.shape[1]-1

    x_train = train_fold.drop(train_fold.columns[class_idx], axis=1)
    y_train = train_fold.iloc[:,class_idx]
    y_train.replace(to_replace=[negative, positive], value=[0,1], inplace=True)

    x_val = test_fold.drop(test_fold.columns[class_idx], axis=1)
    y_val = test_fold.iloc[:,class_idx]
    y_val.replace(to_replace=[negative, positive], value=[0,1], inplace=True)

    d_train = xgb.DMatrix(x_train, y_train)
    d_test = xgb.DMatrix(x_val, y_val)

    params = {'booster': 'gbtree',
        'objective': "binary:logistic",
        'nthread': 4,
        'eval_metric': metric}

    myfit = xgb.train(params = params,
        dtrain = d_train,
        num_boost_round = 100,
        maximize = True)

    return (myfit.predict(d_test))

def acc_per_class(y_true, prediction, class1, class2, classThreshold):
    div_01 = sum(np.logical_and(y_true == class1, 
        np.logical_or(prediction > classThreshold, 
            prediction < (1 - classThreshold)))) 
    div_02 = sum(np.logical_and(y_true == class2, 
        np.logical_or(prediction > classThreshold, 
            prediction < (1 - classThreshold)))) 

    n_class_1 = sum(np.logical_and(y_true == class1, prediction < (1 - classThreshold)))
    n_class_2 = sum(np.logical_and(y_true == class2, prediction > classThreshold))
    acc_class_1 = 0
    acc_class_2 = 0

    if(div_01 > 0):
        acc_class_1 = n_class_1/div_01 *100
    if(div_02 > 0):
        acc_class_2 = n_class_2/div_02 *100
        
    return (acc_class_1, acc_class_2, n_class_1, n_class_2)

def auc_per_class(y_test, y_pred, positive = 1):
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=positive)
    return metrics.auc(fpr, tpr)

def accuracy_per_class(y_test, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    acc_class_1 = tp / (tp + fp)
    acc_class_2 = tn / (tn + fn)

    return acc_class_1, acc_class_2

def weighted_mean(x, w):
    return np.dot(x, w)/np.sum(w)

def runSFC(dataset, nrepetitions = 100, class_idx = np.nan, 
            testSize_smaller_class = 10, classThreshold = 0.5, 
            positive = 'Thrombosis', seed=100):

    np.random.seed(seed)
    if(np.isnan(class_idx)):
        class_idx =  dataset.shape[1]-1

    print(f"Running SFC on label index = {class_idx}\n")

    index_by_class = []
    smaller_size = dataset.shape[0]    

    for class_value in np.unique(dataset.iloc[:,class_idx]):
        temp_idx = np.where(dataset.iloc[:,class_idx] == class_value)[0]
        print(f"Class: {class_value} -- Size: {len(temp_idx)}\n")
        index_by_class.append(temp_idx)
        if len(temp_idx) < smaller_size:
            smaller_size = len(temp_idx)

    accuracy_vec = np.zeros((nrepetitions, len(index_by_class)))
    n_inst_threshold = np.zeros((nrepetitions, len(index_by_class)))

    negative = np.unique(dataset.iloc[:,class_idx]).tolist()
    negative.remove(positive)
    negative = negative[0]

    print(f"Positive class: {positive} | Negative Class:{negative}")
    f1 = []
    auc = []

    for rep in np.arange(nrepetitions):
        print(f"Analyzing iteration {rep}")

        test_idx = np.random.choice(index_by_class[0], testSize_smaller_class, replace=False)
        train_idx = np.random.choice(np.delete(index_by_class[0],
            np.where(np.isin(index_by_class[0], test_idx))), smaller_size, replace=False)
        

        for next_class in np.arange(start=1, stop=len(index_by_class)):
            temp_test_idx = np.random.choice(index_by_class[next_class], testSize_smaller_class, replace=False)            
            test_idx = np.concatenate((test_idx, temp_test_idx))
            train_idx = np.concatenate((train_idx, 
                np.random.choice(np.delete(index_by_class[next_class], 
                np.where(np.isin(index_by_class[0], temp_test_idx))), smaller_size, replace=False)))

        
        if not set(train_idx).isdisjoint(set(test_idx)):
            print(f'Same indexes {rep}')
            print(f'train: {train_idx}')
            print(f'test: {test_idx}')
            [[print(f"i:{i} - {train_idx[test_idx[i] == train_idx]}") if sum(test_idx[i] == train_idx) > 0 else print("")] for i in np.arange(len(test_idx))]
        
        train_set = dataset.iloc[train_idx, :]
        test_set = dataset.iloc[test_idx, :]

        result_by_att = np.zeros((test_set.shape[0], test_set.shape[1]-1))
        #fitting a model per attribute
        for att in np.delete(np.arange(train_set.shape[1]), class_idx):
            result_by_att[:, att] = runXGBoost(train_set.iloc[:, [att, class_idx]], 
                                    test_set.iloc[:, [att, class_idx]], positive, negative)     

        statistical_weights = get_statistical_weights(train_set)
        prediction = np.apply_along_axis(weighted_mean, 1, result_by_att, w = statistical_weights)

        y_pred = np.repeat(0, test_set.shape[0])
        y_pred[np.where(prediction > classThreshold)] = 1
        y_true = test_set.iloc[:,class_idx].replace(to_replace=[negative, positive], value=[0,1])
        ## select the accuracy equation
        ## OP1 - Traditional threshold
        acc_class_1, acc_class_2 = accuracy_per_class(y_true, y_pred)
        n_c1 = n_c2 = 0
        ## OP2 - Symmetric Threshold
        #cc_class_1, acc_class_2, n_c1, n_c2 = acc_per_class(y_true, prediction, 0, 1, classThreshold)
        print(f"\tTrue: {Counter(y_true)}")
        print(f"\tPred: {Counter(y_pred)}")

        f1.append(f1_score(y_true, y_pred, average='macro'))
        auc.append(auc_per_class(y_true, y_pred))

        accuracy_vec[rep, :] = np.nan_to_num(np.array((acc_class_1, acc_class_2)))
        n_inst_threshold[rep, :] = np.array((n_c1, n_c2))

    return accuracy_vec, f1, n_inst_threshold, auc#, temp1, temp2, temp3

                    

