import scipy as sc
from sklearn import cluster, datasets
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import data
train_features = pd.read_csv("train_features.csv") 

test_features = pd.read_csv("test_features.csv") 

# Merge train and test data, and remove the first 4 columns
total_data = [train_features, test_features]
total_data = pd.concat(total_data)#.head(n)
total_data = np.array(total_data)
total_data = total_data[:,4:]


# Here, I used the agglemerative clustering
cluster = AgglomerativeClustering(distance_threshold=np.inf, n_clusters=None, affinity='euclidean', linkage='average')
cluster.fit_predict(total_data)      

link_M = cluster.children_

n = len(total_data)

# This function takes the linkage_Matrix, and creates a system of nested list that represent the hierarchy of the data.
def expand(link_M, m, n):
    
    if m >= n:
        # if true, expand m
        cluster = link_M[m-n] # m-n is the index where cluster "m" is generated
        expanded_cluster = []
        for i in cluster:
            expanded_cluster.append(expand(link_M, i, n))
        return expanded_cluster
    return m
    
dend = expand(link_M, len(link_M)+n-1, n)




    




    