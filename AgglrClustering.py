
import numpy as np
import matplotlib.pyplot as plt

import time

features = 2
data_points = 500 # N

# random points
data = np.random.random(size = (data_points, features))

# Gaussian
data1 = np.random.normal(size = (data_points, features))
data2 = np.random.normal(size = (data_points, features))+np.array([-3,3])
data = np.concatenate((data1, data2))

def clusters_distance(data, i, j):
    '''
    i and j will be lists of indices (i.e. row numbers) of the points in the two clusters.
    '''
    mean_i = np.mean(data[i], axis = 0)
    mean_j = np.mean(data[j], axis = 0)
    
    return np.sqrt(np.sum((mean_i - mean_j)**2))

clusters = []
for i in range(len(data)):
    clusters.append([i])

nClusters = 2

past = time.time()

D_matrix = np.zeros(shape = (len(clusters),len(clusters)))
for i in range(len(clusters)):
        for j in range(len(clusters)):
            D_matrix[i,j] = clusters_distance(data, clusters[i], clusters[j])

while len(clusters)> nClusters:
    
    D_matrix = D_matrix + (np.max(D_matrix)+1) * np.eye(len(clusters))
    clusters_to_merge = np.where(D_matrix == np.min(D_matrix))[0]
    
    clusters.append(clusters[clusters_to_merge[0]] + clusters[clusters_to_merge[1]])
    del clusters[max(clusters_to_merge)]
    del clusters[min(clusters_to_merge)]
    
    D_matrix = np.delete(D_matrix, max(clusters_to_merge), 0)
    D_matrix = np.delete(D_matrix, min(clusters_to_merge), 0)
    D_matrix = np.delete(D_matrix, max(clusters_to_merge), 1)
    D_matrix = np.delete(D_matrix, min(clusters_to_merge), 1)
    
    new_D_matrix = np.zeros((len(clusters),len(clusters)))
    new_D_matrix[:-1,:-1] = D_matrix
    
    for i in range(len(clusters)):
        new_D_matrix[i,-1] = clusters_distance(data, clusters[i], clusters[-1])
    new_D_matrix[-1, :] = new_D_matrix[:,-1]
    
    D_matrix = new_D_matrix.copy()

print(time.time()-past)

colors = []
for i in range(len(data)):
    if i in clusters[0]:
        colors.append('red')
    elif i in clusters[1]:
        colors.append('blue')

plt.scatter(data[:,0], data[:,1], color = colors)

