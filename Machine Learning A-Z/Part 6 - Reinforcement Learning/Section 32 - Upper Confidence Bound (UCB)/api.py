#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 19:25:31 2017

@author: yoovrajshinde
"""

from flask import Flask, request
from flask_restful import Resource, Api, reqparse
from json import dumps
from flask.ext.jsonpify import jsonify

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

DATA = [[0]*5 for i in range(1)]
N=1
d=5
# Implementing the UCB algorithm
number_of_selections = [0] * d
sums_of_rewards = [0] * d
ads_selected = []
total_reward = 0
ad = 0


app = Flask(__name__)
api = Api(app)

class Add(Resource):
    def get(self):
        global N, ad, DATA
        parser = reqparse.RequestParser()
        parser.add_argument('id', type=int)
        args = parser.parse_args()
        clickedId = args['id']
        print("id=", clickedId)
        newData = [0 for x in range(5)]
        newData[clickedId] = 1
        DATA.append(newData)
        N = N + 1
        calculate_optimum_add()
        return {'id':ad}
api.add_resource(Add, '/click')



def calculate_optimum_add():
    global N, d, DATA, number_of_selections, sums_of_rewards, ads_selected, total_reward, ad
        # loop for each round
    for n in range(0, N):
        max_upper_bound = 0
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
        print(DATA)
        print(n)
        print(ad)
        print(type(DATA))
        reward = DATA[n,ad]
        sums_of_rewards[ad] = sums_of_rewards[ad] + reward
        total_reward = total_reward + reward
    
    
    plt.hist(ads_selected)
    plt.title('Histogram of ads selection')
    plt.xlabel('ads')
    plt.ylabel('number of times each ad was selected')
    plt.show()

if __name__ == '__main__':
     app.run(port='5002')

