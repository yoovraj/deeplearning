#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 16:26:56 2017

@author: yoovrajshinde
"""

# importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from statistics import mode, mean
import datetime

## importing the dataset
dataset = pd.read_csv('export-EtherPrice.csv')
dataset.iloc[:,0].values

day, month, year = (int(x) for x in dt.split('/')) 