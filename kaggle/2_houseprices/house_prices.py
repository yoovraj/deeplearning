#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 19:16:34 2017

@author: yoovrajshinde
"""

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## importing the dataset
dataset = pd.read_csv('train.csv')
print("Checking NaN columns")
print(dataset.isnull().any())
