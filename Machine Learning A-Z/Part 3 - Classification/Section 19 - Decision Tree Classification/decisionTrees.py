#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 12:31:46 2017

@author: yoovrajshinde
"""

## Classification Template

# Decision Tree Classifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset = pd.read_csv('Social_Network_Ads.csv')
# to make the X as matrix
X = dataset.iloc[:, [2,3]].values
Y = dataset.iloc[:, 4].values

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
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
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
plt.title('Decision Tree Classifier (training set')
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
plt.title('Decision Tree Classifier (test set')
plt.xlabel('age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()