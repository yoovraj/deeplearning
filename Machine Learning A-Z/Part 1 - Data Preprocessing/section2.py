#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 10:54:13 2017

@author: yoovrajshinde
"""

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
# specify first the working directory

print("Hello worlds")

dataset = pd.read_csv("Data.csv")
print(dataset)


X = dataset.iloc[:, :-1].values
X1 = pd.DataFrame(data=X[0:,0:], index=X[0:,0], columns=X[0,0:])
print(X)
Y = dataset.iloc[:,3].values
print(Y)

# missing data
from sklearn.preprocessing import Imputer
inputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
print(inputer)
inputer = inputer.fit(X[:,1:3])
X[:,1:3] = inputer.transform(X[:,1:3])
print(X)

# categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

onehotencoder = OneHotEncoder(categorical_features= [0])
X = onehotencoder.fit_transform(X).toarray()
X1 = pd.DataFrame(data=X[0:,0:], index=X[0:,0], columns=X[0,0:])
print(X)

labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
print(Y)


## splitting data into training and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state= 0)

## feature scaling
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
