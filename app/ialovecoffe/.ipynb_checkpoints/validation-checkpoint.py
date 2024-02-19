from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import jaccard_score
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_curve

def positive_negative_rate(y_test, y_pred):
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn+fp)
    sensibivity = tp / (tp+fn)

    return round(sensibivity, 5), round(specificity, 5)  


def accuracy_per_class(y_test, y_pred):
    print(y_test)
    print(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    acc_class_1 = tp / (tp + fp)
    acc_class_2 = tn / (tn + fn)

    # Calculates tpr and fpr
    #tpr =  tp/(tp + fn) # sensitivity - true positive rate
    #fpr = 1 - tn/(tn+fp) # 1-specificity - false positive rate
    
    return acc_class_1, acc_class_2


def computer_scores(y_test, y_pred, y_pred_prob, names=False):
    sen, spe = positive_negative_rate(y_test, y_pred)
    f1 = round(f1_score(y_test, y_pred, average='macro'), 5)
    roc = round(roc_auc_score(y_test, y_pred, average='weighted'), 5)
    jac = round(jaccard_score(y_test, y_pred, average='weighted'), 5)
    fmi = round(fowlkes_mallows_score(y_test, y_pred), 5)
    mcc = round(matthews_corrcoef(y_test, y_pred), 5)
    fpr, tpr, thresh = roc_curve(y_test, y_pred_prob[:,1], pos_label=1)

    if names:
        return {'SEN': sen, 'SPE': spe, 'F1':f1, 
                'ROC': roc, 'IOU': jac, 'FMI': fmi, 'MCC': mcc}

    return sen, spe, f1, roc, jac, fmi, mcc, fpr, tpr, thresh

def computer_scores_outlier(y_test, y_pred, names=False):
    sen, spe = positive_negative_rate(y_test, y_pred)
    f1 = round(f1_score(y_test, y_pred, average='macro'), 5)
    roc = round(roc_auc_score(y_test, y_pred, average='weighted'), 5)
    jac = round(jaccard_score(y_test, y_pred, average='weighted'), 5)
    fmi = round(fowlkes_mallows_score(y_test, y_pred), 5)
    mcc = round(matthews_corrcoef(y_test, y_pred), 5)
    fpr, tpr, thresh = [], [], []

    if names:
        return {'SEN': sen, 'SPE': spe, 'F1':f1, 
                'ROC': roc, 'IOU': jac, 'FMI': fmi, 'MCC': mcc}

    return sen, spe, f1, roc, jac, fmi, mcc, fpr, tpr, thresh

