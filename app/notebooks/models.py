import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope
from scipy.spatial import distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def RSNN(X_train, y_train, X_test, y_test, prob=False):
    params = {'n_neighbors':np.arange(1, 25, 1), 
              'weights':np.array(['uniform', 'distance']),
              'p':np.arange(1, 4, 1)
             }    
   
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    model = KNeighborsClassifier()
    model_cv = RandomizedSearchCV(model, params, cv = cv, scoring='f1_macro', n_jobs = -1, n_iter=50)

    model_cv.fit(X_train, y_train)
    
    if prob:
        y_pred = model_cv.predict_proba(X_test)
    else:
        y_pred = model_cv.predict(X_test)
    
    return model_cv, y_pred


def RSLocalOutlierFactor(X_train, y_train, X_test, y_test, measure='f1_macro'):
    
    pipe = Pipeline([
        ("clf", LocalOutlierFactor(novelty=True))
    ])

    params = {'clf__algorithm':['ball_tree', 'kd_tree', 'brute'], 
              'clf__n_neighbors':[5, 10, 15, 20, 25, 30, 35, 40, 45, 50], 
              'clf__contamination': [0.35, 0.40, 0.45, 0.5], 
              'clf__leaf_size': [5, 10, 20, 30, 40],
              'clf__metric': ['cityblock', 'euclidean', 'l1', 'l2', 'manhattan'], 
              'clf__p':[1,2,3,4,5]}

    model_cv = RandomizedSearchCV(pipe, params, n_iter=1000, 
                                  scoring=measure, 
                                  n_jobs = -1, 
                                  random_state=1)

    model_cv.fit(X_train, y_train)
    y_pred = model_cv.predict(X_test)
    
    return model_cv, y_pred

def RSDT(X_train, y_train, X_test, y_test, prob=False):
    params = {'max_depth':np.arange(1, 25, 1), 
              'criterion':np.array(['gini', 'entropy']),
              'min_samples_leaf':np.arange(1, 25, 1)
             }    
   
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    model = DecisionTreeClassifier()
    model_cv = RandomizedSearchCV(model, params, cv = cv, scoring='f1_macro', n_jobs = -1, n_iter=50)

    model_cv.fit(X_train, y_train)
    
    if prob:
        y_pred = model_cv.predict_proba(X_test)
    else:
        y_pred = model_cv.predict(X_test)
    
    return model_cv, y_pred

def RSRF(X_train, y_train, X_test, y_test, prob=False):
    params = {'n_estimators':np.arange(1, 25, 1), 
              'max_depth':np.arange(1, 25, 1)
             }    
   
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    model = RandomForestClassifier(random_state=42)
    model_cv = RandomizedSearchCV(model, params, cv = cv, scoring='f1_macro', n_jobs = -1, n_iter=50)

    model_cv.fit(X_train, y_train)
    
    if prob:
        y_pred = model_cv.predict_proba(X_test)
    else:
        y_pred = model_cv.predict(X_test)
    
    return model_cv, y_pred