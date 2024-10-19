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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

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
X = df[['X', 'Y', 'Z']] 
Y = df['Step']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2,random_state = 42)

#Logistic Regression Model
logistic_regression = LogisticRegression(max_iter=10000)
parametric_grid_lr = { 'C': [0.01, 0.1, 1, 10, 100]
    }
grid_search_lr = GridSearchCV(logistic_regression, parametric_grid_lr, cv=5)
grid_search_lr.fit(X_train, y_train)
best_model_lr = grid_search_lr.best_estimator_
Y_pred_lr = grid_search_lr.predict(X_test)
accuracy_lr = accuracy_score(y_test, Y_pred_lr)
precision_lr = precision_score(y_test, Y_pred_lr, average='weighted')
print("Best Logistic Regression Model:", best_model_lr)

#Random Forest Classifier (with RandomizedSearchCV)
random_forest = RandomForestClassifier()
parametric_grid_rf = {'n_estimators': [10, 30, 50, 100],
'max_depth': [None, 10, 20, 30],
'min_samples_split': [2, 5, 10],
    }
grid_search_rf = RandomizedSearchCV(random_forest, parametric_grid_rf, cv=5, scoring='accuracy', n_jobs=1)
grid_search_rf.fit(X_train, y_train)
best_model_rf = grid_search_rf.best_estimator_
Y_pred_rf = grid_search_rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, Y_pred_rf)
precision_rf = precision_score(y_test, Y_pred_rf, average='weighted')
print("Best Random Forest Model:", best_model_rf)

#Support Vector Machine
svm = SVC()
parametric_grid_svm = {'kernel': ['linear', 'rbf'],
  'C': [0.1, 1, 10, 100],
  'gamma': ['scale', 'auto']
    }
grid_search_svm = GridSearchCV(svm, parametric_grid_svm, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_svm.fit(X_train, y_train)
best_model_svm = grid_search_svm.best_estimator_
Y_pred_svm = grid_search_svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, Y_pred_svm)
precision_svm = precision_score(y_test, Y_pred_svm, average='weighted')
print("Best Support Vector Machine Model:", best_model_svm)


#Step 5 - Model Performance Analysis
#F1 Scores
f1_lr = f1_score(y_test, Y_pred_lr, average='weighted')
f1_rf = f1_score(y_test, Y_pred_rf, average='weighted')
f1_svm = f1_score(y_test, Y_pred_svm, average='weighted')

#Choose best Model based on F1 Score, Precision and Accuracy
print(f"Logistic Regression F1 Score: {f1_lr}")
print(f"Logistic Regression Test Accuracy: {accuracy_lr}")
print(f"Logistic Regression Test Precision: {precision_lr}")
print(f"Random Forest F1 Score: {f1_rf}")
print(f"Random Forest Test Accuracy: {accuracy_rf}")
print(f"Random Forest Test Precision: {precision_rf}")
print(f"Support Vector Machine F1 Score: {f1_svm}")
print(f"Support Vector Machine Accuracy: {accuracy_svm}")
print(f"Support Vector Machine Precision: {precision_svm}")

#Confusion Matricies
confusion_matrix_lr = confusion_matrix(y_test, Y_pred_lr)
confusion_matrix_rf = confusion_matrix(y_test, Y_pred_rf)
confusion_matrix_svm = confusion_matrix(y_test, Y_pred_svm)

def plot_confusion_matrix(matrix, title):
    plt.figure(figsize=(6,4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap='Blues', cbar=False)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

plot_confusion_matrix(confusion_matrix_lr, 'Logistic Regression Confusion Matrix')
plot_confusion_matrix(confusion_matrix_rf, 'Random Forest Confusion Matrix')
plot_confusion_matrix(confusion_matrix_svm, 'Support Vector Machine Confusion Matrix')












