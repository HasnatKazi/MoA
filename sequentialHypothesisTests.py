assert False, 'This line is to stop spyder from executing the entire script'

import scipy as sc
from sklearn import cluster, datasets
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import data
train_features = pd.read_csv("train_features.csv")
test_features = pd.read_csv("test_features.csv")

n_train = train_features.shape[0]
n_test = test_features.shape[0]

# Merge train and test data, and remove the first 4 columns
total_data = [train_features, test_features]
total_data = pd.concat(total_data)#.head(n)
total_data = np.array(total_data)
total_data = total_data[:,4:]

n_total = total_data.shape[0]

target = np.array(pd.read_csv("train_targets_scored.csv"))[:n_train,1:]
#target = target[:,np.where(target.sum(axis = 0) >0)[0]]

n_targets = target.shape[1]

p_global = target.sum(axis = 0)/n_train
p_sd = (p_global*(1-p_global))**0.5

# Here, I used the agglemerative clustering
cluster = AgglomerativeClustering(distance_threshold=np.inf, n_clusters=None, affinity='euclidean', linkage='average')
cluster.fit_predict(total_data)      

link_M = cluster.children_

cluster_counts = {i: [target[i],1] for i in range(target.shape[0])}
cluster_counts.update({i:[np.zeros(n_targets),0] for i in range(n_train, n_total)})

for i, (u, v) in enumerate(link_M):
    sum_of_counts = cluster_counts[u][0] + cluster_counts[v][0]
    sum_of_sample_size = cluster_counts[u][1] + cluster_counts[v][1]
    new_cluster = [sum_of_counts, sum_of_sample_size]
    cluster_counts[i + n_total] = new_cluster

cluster_p_hat = {i: v[0]/v[1] for i, v in cluster_counts.items() if v[1]>0 and i>=n_total}
cluster_z_val = {i: np.abs(v[0]/v[1]-p_global)*np.sqrt(v[1])/p_sd for i, v in cluster_counts.items() if v[1]>0 and i>=n_total}

max_z_vals = np.asarray([[k,v.max()] for k, v in cluster_z_val.items()])
plt.plot(max_z_vals[:,0],max_z_vals[:,1], '.', markersize = 1)

plt.hist(max_z_vals[:,1], bins = 100)

MoA = 0
MoA_z_vals = np.asarray([[k,v[MoA]] for k, v in cluster_z_val.items()])
plt.plot(MoA_z_vals[:,0],MoA_z_vals[:,1], '.', markersize = 1)

p_val_thr = 0.05/len(max_z_vals)/n_targets

from scipy.stats import norm, binom_test

z_val_thr = norm.ppf(1-p_val_thr)

MoA = 2
MoA_z_vals = np.asarray([[k,v[MoA]] for k, v in cluster_z_val.items()])

significant_z_vals = MoA_z_vals[MoA_z_vals[:,1]>z_val_thr]

plt.plot(significant_z_vals[:,0],significant_z_vals[:,1], '.', markersize = 1)

def multiple_bin_test(count, total, p_global, k = 0):
    return binom_test(count[k], total, p_global[k])
    return np.array([binom_test(i, total, p) for i, p in zip(count, p_global)])

MoA = 20

cluster_bin_p_val = [[i, multiple_bin_test(v[0], v[1], p_global, MoA)]
                     for i, v in cluster_counts.items()
                     if v[1]>0 and i>=n_total]
cluster_bin_p_val = np.asarray(cluster_bin_p_val)

plt.plot(-np.log(cluster_bin_p_val[:,1]), '.', markersize = 1)
plt.hlines(-np.log(p_val_thr), 0, len(cluster_bin_p_val))
