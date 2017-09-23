#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 15:49:11 2017

@author: yoovrajshinde
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')


# random selection
import random
N = 10000
d = 10
ads_selected = []
total_reward = 0
for n in range(0,N):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    total_reward = total_reward + reward


plt.hist(ads_selected)
plt.title('Histogram of ads selection')
plt.xlabel('ads')
plt.ylabel('number of times each ad was selected')
plt.show()



# Implementing the UCB algorithm
number_of_selections = [0] * d
sums_of_rewards = [0] * d
ads_selected = []
total_reward = 0
N=10000
d=10

# loop for each round
for n in range(0, N):
    max_upper_bound = 0
    ad = 0
    # loop over all versions of add
    for i in range(0,d):
        
        # apply strategy after 1st 10 rounds
        if (number_of_selections[i] > 0):
            average_reward = sums_of_rewards[i] / number_of_selections[i]
            
            # log takes from 1 while python indexes are from 0, so n+1
            delta_i = math.sqrt(3/2 * math.log(n+1) / number_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        # selecting add which has max upper bound
        if (upper_bound > max_upper_bound) :
            max_upper_bound = upper_bound
            ad = i
            
    # set the selected add
    ads_selected.append(ad)
    
    # update the number of selections and sums of rewards
    number_of_selections[ad] =  number_of_selections[ad] + 1
    reward = dataset.values[n,ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward


plt.hist(ads_selected)
plt.title('Histogram of ads selection')
plt.xlabel('ads')
plt.ylabel('number of times each ad was selected')
plt.show()

"""
N = {}
R = {}
D = {}
r = {}
for n in range(N):
    for i in range(10):
        
        # add selection strategy
        i = n
        # calculate number of times the ad was selected till n
        N[i] = (N.get(i) or 0) + 1;
        # sum of rewards of add i upto round n 
        R[i] = (R.get(i) or 0) + dataset.values[n,i]
        r[i] = R[i]/N[i]
        D[i] = 1.5*math.log(n+1)/N[i] 
        UCB = r[i] + D[i]
"""