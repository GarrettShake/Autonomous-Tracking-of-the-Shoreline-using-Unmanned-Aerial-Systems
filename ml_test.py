# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 14:28:31 2019

@author: gasha
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import time


def Build_Data_Set():
    data = pd.read_csv("bhv1.csv")
    X = data[ [
#               'Laplacian Mean',
#               'Laplacian Standard Deviation',
#               'Canny Mean',
#               'Canny Standard Deviation',
#               'Hue Mean',
#               'Blue Mean',
#               'Green Mean',
#               'Red Mean',
               'Shannons Entropy'
               ] ].values

    Y = data["Class"].values
    X= MinMaxScaler().fit_transform(X)
    return X,Y

def Analysis():


    X, y = Build_Data_Set()
#    print(y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
   
#    clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
#    clf = GaussianNB()
#    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(9, 2), random_state=1)
#    clf = DecisionTreeClassifier(random_state=0)
    clf = svm.SVC(kernel="sigmoid", C= 1.0,gamma='auto')
    t0 = time.time()
    clf.fit(X_train,y_train)
    t1 = time.time()

    print("Accuracy:", accuracy_score(y_test,clf.predict(X_test)))
    print(confusion_matrix(y_test,clf.predict(X_test)))
    print(classification_report(y_test,clf.predict(X_test),target_names=['water','land']))
    print(precision_recall_fscore_support(y_test,clf.predict(X_test),average='macro'))
    print("runtime:",round(t1-t0,3))
    
Analysis()
