#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 14:06:21 2017

@author: yoovrajshinde
"""

# conda install -c conda-forge keras
# conda create -n tensorflow python=3.5
# pip install  --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.3.0-py3-none-any.whl


# Logistic Regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset = pd.read_csv('Churn_Modelling.csv')
# to make the X as matrix
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X =LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
labelencoder_X_gen =LabelEncoder()
X[:, 2] = labelencoder_X_gen.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]
# Splitting data into training and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# part 2 ANN model
import keras
from keras.models import Sequential
from keras.layers import Dense
keras.__version__

# Initilizing the ANN 
classifier = Sequential() 

# Adding the input layer and the first hidden layer

# average of input and output  = 11 + 1 / 2
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation='relu', input_dim = 11))

# no need to specify input_dim because the network knows what to expect
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation='relu'))

# output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation='sigmoid'))

# compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# fitting the ANN to training set
classifier.fit(X_train, Y_train, batch_size=10, epochs=100)

## predicting a new result with logistic regression
Y_pred = classifier.predict(X_test)
Y_pred[:,0]
# Visualizing the polynomial regression
plt.plot(Y_test, color = 'red')
plt.plot(Y_pred, color = 'blue')
plt.plot(Y_pred[:,0]-Y_test, color = 'black')
plt.show()



# make a confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred[:,0])
