#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 14:55:37 2017

@author: yoovrajshinde
"""

## Regression Template

# Polynomial Regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
# to make the X as matrix
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values


# Splitting data into training and testing data
"""from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y)

# Fitting SVR model to the dataset
# Create a regressor here
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, Y)

## predicting a new result with polynomial regression
Y_pred = sc_Y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

# Visualizing the polynomial regression
plt.scatter(X, Y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title("Turth or Bluff (SVR model)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()



# Visualizing the polynomial regression for higher resolution and smoother curve
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title("Turth or Bluff (Regressio model)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

