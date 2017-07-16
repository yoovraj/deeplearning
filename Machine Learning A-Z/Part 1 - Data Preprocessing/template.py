#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 15:26:11 2017

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

# separate dependant and independant variables
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:,3].values
print(X)
print(Y)


## splitting data into training and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state= 0)

## feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)"""

