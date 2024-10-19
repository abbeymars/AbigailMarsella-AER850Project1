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
import joblib as jb

#Step 1 - Read Data File
df = pd.read_csv("Project_1_Data.csv")
print(df.info())

#Step 2 - Visualize Data
x = df['X'].values
y = df['Y'].values
z = df['Z'].values
Step = df['Step'].values

plt.plot(Step, x, label= 'X')
plt.plot(Step, y, label= 'Y')
plt.plot(Step, z, label= 'Z')


plt.xlabel('Step')
plt.ylabel('Values')
plt.title('Line Plot')
plt.legend()
plt.show()