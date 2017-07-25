#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 19:13:31 2017

@author: yoovrajshinde
"""

# Simple Linear Regressor
# importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
## X is matrix of features
X = dataset.iloc[:, :-1].values

## Y is vector
Y = dataset.iloc[:, 1].values

## splitting the dataset into training and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Feature scaling 
# not required
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
"""

# Fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the test results
y_pred = regressor.predict(X_test)

# view the training set results
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title("Salary vs Experience (training set)")
plt.xlabel("years of experience")
plt.ylabel("Salary")
plt.show()

plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_test, y_pred, color = 'green')
plt.show()