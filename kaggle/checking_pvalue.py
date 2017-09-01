#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 20:28:50 2017

@author: yoovrajshinde
"""

# importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## importing the dataset
dataset = pd.read_csv('train.csv')
## X is matrix of features
X = dataset.iloc[:, [2,4,5,6]].values

## Y is vector
Y = dataset.iloc[:, 1].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X)
X = imputer.transform(X)




# Splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting Classifier model to the dataset
# Create a classifier here
#from sklearn.linear_model import LogisticRegression
#classifier = LogisticRegression(random_state=0)


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=15, metric='minkowski', p=2)
classifier.fit(X_train, Y_train)



## predicting a new result with logistic regression
Y_pred = classifier.predict(X_test)

# Visualizing the polynomial regression
plt.plot(Y_test, color = 'red')
plt.plot(Y_pred, color = 'blue')
plt.plot(Y_pred-Y_test, color = 'black')
plt.show()



# make a confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)





# Building the optimal model with backward elimination method
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((891,1)).astype(int), values = X, axis=1)

# initialize with all the variables (all - in)
# [C,  0, 2,4,5,6,7,9]
X_opt = X[:,[0, 1, 2, 3, 4, 5, 6, 7]]

# select significance level to stay in the model SL=0.05
# 
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()


# [2,4,5,6]
X_opt = X[:,[0, 1, 2, 3, 4]]# 
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()
