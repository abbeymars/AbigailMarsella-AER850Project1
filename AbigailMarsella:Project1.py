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
parametric_grid_lr = {
    }
grid_search_lr = GridSearchCV(linear_regression, parametric_grid_lr, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search_lr.fit(X_train, y_train)
best_model_lr = grid_search_lr.best_estimator_
Y_pred_lr = grid_search_lr.predict(X_test)
accuracy_lr = accuracy_score(y_test, Y_pred_lr)
print("Best Linear Regression Model:", best_model_lr)
print(f"Linear Regression Test Accuracy: {accuracy_lr}")

#Random Forest Classifier (with RandomizedSearchCV)
random_forest = RandomForestClassifier()
parametric_grid_rf = {
    }
grid_search_rf = RandomizedSearchCV(random_forest, param_distributions=parametric_grid_rf, cv=5, scoring='neg_mean_absolute_error', n_jobs=1)
grid_search_rf.fit(X_train, y_train)
best_model_rf = grid_search_rf.best_estimator_
Y_pred_rf = grid_search_rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, Y_pred_rf)
print("Best Random Forest Model:", best_model_rf)
print(f"Random Forest Test Accuracy: {accuracy_rf}")

#Support Vector Machine
svm = SVR()
parametric_grid_svm = {
    }
grid_search_svm = GridSearchCV(svm, parametric_grid_svm, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search_svm.fit(X_train, y_train)
best_model_svm = grid_search_svm.best_estimator_
Y_pred_svm = grid_search_svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, Y_pred_svm)
print("Best Support Vector Machine Model:", best_model_svm)
print(f"Support Vector Machine Accuracy: {accuracy_rf}")



