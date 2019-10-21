# (1011 HOMEWORK)隨機森林或Adaboost  對參數進行分析

#DecisionTree
#import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing,tree
from sklearn.cross_validation import train_test_split
#using sklearn training adaboost
from sklearn.metrics import accuracy_score

avocado=pd.read_csv("avocado.csv")

x=pd.DataFrame([avocado["Total Volume"],
                avocado["Total Bags"],
                avocado["AveragePrice"],
                avocado["Small Bags"],
                avocado["Large Bags"],
                avocado["XLarge Bags"],]).T


y=avocado["type"]

#切割成75%訓練集，25%測試集
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=1)

#決策樹分類器進行分類
dtree=tree.DecisionTreeClassifier(criterion='entropy',max_depth=4,random_state=0)

#using sklearn training adaboost
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC

#決策分類器性能

clf = dtree.fit(X_train, y_train)
y_train_pred = dtree.predict(X_train)
y_test_pred = dtree.predict(X_test)
tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print('Decision tree train/test accuracies %.3f/%.3f' % (tree_train, tree_test))

# Boosting分類器性能
ada = AdaBoostClassifier( n_estimators=1000, learning_rate=0.1, random_state=0)
ada = ada.fit(X_train, y_train)
y_train_pred = ada.predict(X_train)
y_test_pred = ada.predict(X_test)
ada_train = accuracy_score(y_train, y_train_pred)
ada_test = accuracy_score(y_test, y_test_pred)
print('AdaBoost train/test accuracies %.3f/%.3f' % (ada_train, ada_test)) 
