#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 12:02:03 2024

@author: abbeymarsella
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score

import joblib as jb

#Step 1 - Read Data File
df = pd.read_csv("Project_1_Data.csv")
print(df.info())


#Step 2 - Visualize Data
#arrays
x = df['X'].values
y = df['Y'].values
z = df['Z'].values
Step = df['Step'].values
#plots 
plt.plot(df['Step'], df['X'], label='X Data')
plt.plot(df['Step'], df['Y'], label='Y Data')
plt.plot(df['Step'], df['Z'], label='Z Data')
#lables and legends
plt.xlabel('Step')
plt.ylabel('Points')
plt.title('X, Y, Z Data Points vs Step')
plt.legend()
plt.show()


#Step 3 - Correlation Analysis
#Using Pearson Correlation
corr_matrix = df.corr()
sns.heatmap(np.abs(corr_matrix))


#Step 4 - Classification Model Development and Engineering
#split
X = ['X','Y','Z'] #feature matrix
Y = ['Step'] #target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state = 42)

#Linear Regression Model
linear_regression = LinearRegression()
parametric_grid_lr = {}
grid_search_lr = GridSearchCV(linear_regression, parametric_grid_lr, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search_lr.fit(X_train, y_train)
best_model_lr = grid_search_lr.best_estimator_
Y_pred_log = grid_search_lr.predict(X_test)
accuracy_log = accuracy_score(y_test, Y_pred_log)
print("Best Linear Regression Model:", best_model_lr)
print(f"Linear Regression Test Accuracy: {accuracy_log:.2f}")


#Random Forest Classifier (with RandomizedSearchCV)
CV_model = RandomForestClassifier()
grid_search = RandomizedSearchCV(CV_model, param_distributions=param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=1)
grid_search.fit(X_train, Y_train)
Y_pred_CV = grid_search.predict(X_test)

best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)
best_modelCV = grid_search.best_estimator_


accuracy_CV = accuracy_score(Y_test, Y_pred_CV)
print(f"CV Accuracy: {accuracy_CV}")





#Support Vector Machine
svr = SVR()
param_grid_svr = {
    'kernel': ['linear', 'rbf'],
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto']
}
grid_search_svr = GridSearchCV(svr, param_grid_svr, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search_svr.fit(X_train, y_train)
best_model_svr = grid_search_svr.best_estimator_
print("Best SVM Model:", best_model_svr)



