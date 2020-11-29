import scipy as sc
from sklearn import cluster, datasets
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as ss

from itertools import combinations, product

train_drug = pd.read_csv("train_drug.csv") 

train_drug = np.array(train_drug)

ids = np.arange(23814)

def get_links(drug):
    idx = ids[train_drug[:,1] == drug]
    mustLink = list(combinations(idx, 2))
    return mustLink

uniques, counts = np.unique(train_drug[:,1], return_counts=True)

mustLink = []
for drug in uniques:
    mustLink += get_links(drug)
    
test_ids = np.arange(23814, 23814+3982)

print('Almost finished mustLink.')

mustLink += list(product(test_ids, np.arange(23814)))

print('finished mustLink.')

sm = ss.coo_matrix((len(ids), len(ids)), np.int32).tocsr()

for i in np.arange(len(mustLink)): # add links to both sides of the matrix
    sm[mustLink[i, 0], mustLink[i, 1]] = 1
    sm[mustLink[i, 1], mustLink[i, 0]] = 1
for i in np.arange(sm.tocsr()[1].shape[1]): # add diagonals
    sm[i,i] = 1
sm = sm.tocoo()

train_features = pd.read_csv("train_features.csv") 
test_features = pd.read_csv("test_features.csv") 

# Merge train and test data, and remove the first 4 columns
total_data = [train_features, test_features]
total_data = pd.concat(total_data)#.head(n)
total_data = np.array(total_data)
total_data = total_data[:,4:]

m = AgglomerativeClustering(distance_threshold=np.inf,
                            n_clusters=None,
                            affinity='euclidean',
                            linkage='average',
                            connectivity=sm)
out = m.fit_predict(total_data)
