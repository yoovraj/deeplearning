#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 15:31:02 2017

@author: yoovrajshinde
"""

#%reset -f

#Importing the libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the mall dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values

# Using the dendogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()


# Fitting hierarchical clustering to the mall dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)


# Visualizing the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s= 100, c='red', label='Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s= 100, c='blue', label='Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s= 100, c='green', label='Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s= 100, c='cyan', label='Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s= 100, c='magenta', label='Cluster 5')
plt.title('Cluster of clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()



n_max = 5
color = ('red', 'blue', 'green', 'cyan', 'magenta')
for i in range(1,n_max+1):
    output_file_name = str(i) + ".jpg"
    for j in range(0,i):
        print("i={} j={} ".format(i, j))
    hc = AgglomerativeClustering(n_clusters = i, affinity = 'euclidean', linkage = 'ward')
    y_hc = hc.fit_predict(X)
    for j in range(0,i):
        plt.scatter(X[y_hc == j, 0], X[y_hc == j, 1], s= 100, c=color[j], label=j)
    plt.title('Cluster of clients with number of clusters n={}'.format(i))
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    plt.savefig(output_file_name)
    plt.clf()


import glob
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from moviepy.editor import ImageSequenceClip

# Define pathname to save the output video
output = 'test.mp4'
path = '*.jpg'
output_img_list = glob.glob(path)

#data = Databucket() # Re-initialize data in case you're running this cell multiple times
clip = ImageSequenceClip(output_img_list, fps=1) # Note: output video will be sped up because 
                                          # recording rate in simulator is fps=25
#new_clip = clip.fl_image(process_image) #NOTE: this function expects color images!!
clip.write_videofile(output, audio=False)

