#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 13:43:24 2017

@author: yoovrajshinde
"""

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
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""

# Fitting using Liner regression model
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(X, Y)


# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)

linear_regressor_2 = LinearRegression()
linear_regressor_2.fit(X_poly, Y)


# Visualizing the linear regression results
plt.scatter(X, Y, color = 'red')
plt.plot(X, linear_regressor.predict(X), color = 'blue')
plt.title("Turth or Bluff (Linear Regression)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()


X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)

# Visualizing the polynomial regression
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, linear_regressor_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title("Turth or Bluff (Polynomial Regression)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()



## predicting a new result with linear regression
linear_regressor.predict(6.5)

## predicting a new result with polynomial regression
linear_regressor_2.predict(poly_reg.fit_transform(6.5))