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
    
test_ids = np.arange(23814, 27796)
    
print('Almost finished mustLink.')

mustLink += list(product(test_ids, np.arange(23814)))

print('finished mustLink.')

mustLink = np.array(mustLink)

ones = np.ones(mustLink.shape[0], np.int32)
rows = mustLink[:,0]
columns = mustLink[:,1]

sm = ss.coo_matrix((ones, (rows, columns)), shape=(27796, 27796))

del mustLink

train_features = pd.read_csv("train_features.csv") 
test_features = pd.read_csv("test_features.csv") 

# Merge train and test data, and remove the first 4 columns
total_data = [train_features, test_features]
total_data = pd.concat(total_data)#.head(n)
total_data = np.array(total_data)
total_data = total_data[:,4:]

del train_features
del test_features

m = AgglomerativeClustering(distance_threshold=np.inf,
                            n_clusters=None,
                            affinity='euclidean',
                            linkage='average',
                            connectivity=sm)
out = m.fit_predict(total_data)


from scipy.cluster.hierarchy import dendrogram
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)