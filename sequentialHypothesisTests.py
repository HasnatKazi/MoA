assert False, 'This line is to stop spyder from executing the entire script'

import scipy as sc
from sklearn import cluster, datasets
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

total_features_df = pd.read_csv("train_features.csv")

n_total = total_features_df.shape[0]

n_train = int(n_total*0.8)
n_test = n_total - n_train

total_data = np.array(total_features_df)
total_data = total_data[:,4:]

cluster = AgglomerativeClustering(distance_threshold=np.inf, n_clusters=None,
                                  affinity='euclidean', linkage='complete')
cluster.fit_predict(total_data)

train_data = total_data[:n_train,:]
test_data = total_data[n_train:,:]

assert train_data.shape[0] == n_train, f'train data has {train_data.shape[0]} rows, not {n_train}.'
assert test_data.shape[0] == n_test, f'train data has {test_data.shape[0]} rows, not {n_test}.'

target_df = pd.read_csv("train_targets_scored.csv")
target_data = np.array(target_df)[:,1:]
n_targets = target_data.shape[1]

assert target_df.shape[0] == n_total, f'Target dataframe has {target_df.shape[0]} rows. Required {n_total} rows.'

train_target = target_data[:n_train]
test_target = target_data[n_train:]

p_global = train_target.sum(axis = 0)/n_train
p_sd = (p_global*(1-p_global))**0.5

link_M = cluster.children_
assert len(link_M) == n_total -1, 'Wrong number of clusters.'


cluster_counts_train = [[target_data[i], np.zeros(n_targets), 1, 0] for i in range(n_train)]
cluster_counts_test = [[np.zeros(n_targets), target_data[i], 0, 1] for i in range(n_train, n_total)]
cluster_counts = cluster_counts_train + cluster_counts_test

for u, v in link_M:
    clusters_to_merge = zip(cluster_counts[u], cluster_counts[v])
    new_cluster = [a + b for a, b in clusters_to_merge]    
    cluster_counts.append(new_cluster)

cluster_p_hat = [[c0/n0, c1/n1, n1] for c0, c1, n0, n1 in cluster_counts]

for i, (_, _, n0, _) in enumerate(cluster_counts):
    if n0 == 0:
        cluster_p_hat[i][0] = p_global
        
def log_loss(p_train, p_test, N_test):
    p = np.maximum(np.minimum(p_train.astype(float),1-1e-15),1e-15)
    hidden_p = np.maximum(np.minimum(p_test.astype(float),1-1e-15),1e-15)
    return np.sum( - (hidden_p * np.log(p) + (1-hidden_p) * np.log(1-p)) ) * N_test

cluster_loss = [log_loss(*c) for c in cluster_p_hat]

link_M_extended = np.concatenate((np.array([[None, i] for i in range(n_total)]),  link_M))

assert len(cluster_loss) == len(link_M_extended), 'The number of cluster and the number of mergers do not match'

def loss_after_split(i):
    j, k = link_M_extended[i]
    
    if j == None:
        return np.inf
    
    loss = 0
    
    if not np.isnan(cluster_loss[j]):
        loss += cluster_loss[j]
    if not np.isnan(cluster_loss[k]):
        loss += cluster_loss[k]
        
    return loss

delta = 100

frontier = [-1]
loss_changes = [0]
end_nodes = []

track_loss = [cluster_loss[-1],]

while len(frontier) != 0:
    to_explore = frontier.pop(0)
    loss_changes.pop(0)
    
    if loss_after_split(to_explore) < cluster_loss[to_explore] + delta:
        
        loss_decrease = loss_after_split(to_explore) - cluster_loss[to_explore]
        
        sorted_pos = np.searchsorted(loss_changes, loss_decrease)
        
        for i in link_M_extended[to_explore]:
            if not np.isnan(cluster_loss[i]):
                loss_changes.insert(sorted_pos, loss_decrease)
                frontier.insert(sorted_pos, i)
    else:
        end_nodes.append(to_explore)
        
    current_loss = 0
    for i in frontier:
        current_loss += cluster_loss[i]
    for i in end_nodes:
        current_loss += cluster_loss[i]
    track_loss.append(current_loss)

plt.plot(track_loss)

