#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 19:14:35 2017

@author: yoovrajshinde
"""

# importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

## importing the dataset
dataset = pd.read_csv('train.csv')
dataset.fillna(0)
## X is matrix of features
X_train = dataset.iloc[:, [2,4,5,6,7,9, 11]].values
X_train[[61,829],6] = 'S'

## Y is vector
Y_train = dataset.iloc[:, 1].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X1 = LabelEncoder()
X_train[:, 1] = labelencoder_X1.fit_transform(X_train[:, 1])

labelencoder_X6 = LabelEncoder()
X_train[:, 6] = labelencoder_X6.fit_transform(X_train[:, 6])


# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer = imputer.fit(X_train[:,6])



onehotencoder = OneHotEncoder(categorical_features = [6])
X_train[:,6] = onehotencoder.fit_transform(X_train).toarray()




"""
# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer = imputer.fit(X_train[:,6])
X_train[:,6] = imputer.transform(X_train[:,6])
x = [0,np.nan]

for x in X_train[:,6]:
    if np.isnan(x):
        print ("Found ")

X_train[:,6]
np.isnan(X_train[:,6])
from statistics import mode
mode(X_train[:,6])
"""

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


#from sklearn.neighbors import KNeighborsClassifier
#classifier = KNeighborsClassifier(n_neighbors=15, metric='minkowski', p=2)

from sklearn.tree import DecisionTreeClassifier
regressor = DecisionTreeClassifier(random_state=0)


regressor.fit(X_train, Y_train)



## predicting a new result with logistic regression
Y_pred = regressor.predict(X_test)

# Visualizing the polynomial regression
plt.plot(Y_test, color = 'red')
plt.plot(Y_pred, color = 'blue')
plt.plot(Y_pred-Y_test, color = 'black')
plt.show()



# make a confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)


# Visualizing the Training set result
from matplotlib.colors import ListedColormap
X_set, Y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:,0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:,1].max() + 1, step=0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red','green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i,j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j,0], X_set[Y_set == j,1],
                c = ListedColormap(('red', 'green'))(i), label=j)
plt.title('Logistic Regression (training set')
plt.xlabel('age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()



# Visualizing the Test set result
from matplotlib.colors import ListedColormap
X_set, Y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:,0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:,1].max() + 1, step=0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red','green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i,j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j,0], X_set[Y_set == j,1],
                c = ListedColormap(('red', 'green'))(i), label=j)
plt.title('Logistic Regression (test set')
plt.xlabel('age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()