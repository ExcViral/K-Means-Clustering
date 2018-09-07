# import the necessary libraries

import math

import numpy as np
import pandas as pd

# import the dataset

df = pd.read_csv("iris.csv")

# fetch all features in different sets

id = df['id'].values
f1 = df['sepallengthcm'].values
f2 = df['sepalwidthcm'].values
f3 = df['petallengthcm'].values
f4 = df['petalwidthcm'].values

data = np.array(list(zip(f1, f2, f3, f4)))


# ================================================================================================
# ==== Helper functions ==========================================================================
# ================================================================================================

# function to intialize random centroids at k different locations

def init_centroids(k, data):
    c = []
    s = np.random.randint(low=1, high=len(data), size=k)
    while (len(s) != len(set(s))):
        s = np.random.randint(low=1, high=len(data), size=k)
    for i in s:
        c.append(data[i])
    # c = np.random.rand(k, len(data[0]))
    # c = [list(i) for i in c]
    # print(c)
    return c

# function to calculate euclidean distance between two rows

def euc_dist(a, b):
    sum = 0
    for i, j in zip(a, b):
        a = (i - j) * (i - j)
        sum = sum + a
    return math.sqrt(sum)


# function to generate the distance table for each point w.r.t all centroids

def cal_dist(centroids, data):
    c_dist = []
    for i in centroids:
        temp = []
        for j in data:
            temp.append(euc_dist(i, j))
        c_dist.append(temp)
    return c_dist


# function to perform clustering on the basis of distance table w.r.t to all centroids

def perf_clustering(k, dist_table):
    # create empty cluster list of size k
    clusters = []
    for i in range(k):
        clusters.append([])
    # start clustering data points, such that each point is clustered to nearest centroid
    for i in range(len(dist_table[0])):
        d = []
        for j in range(len(dist_table)):
            d.append(dist_table[j][i])
        clusters[d.index(min(d))].append(i)
    return clusters


# function to update the centroids locations

def update_centroids(centroids, cluster_table, data):
    for i in range(len(centroids)):
        if (len(cluster_table[i]) > 0):
            temp = []
            for j in cluster_table[i]:
                temp.append(list(data[j]))

            sum = [0] * len(centroids[i])
            for l in temp:
                sum = [(a + b) for a, b in zip(sum, l)]

            centroids[i] = [p / len(temp) for p in sum]
    return centroids


# ================================================================================================
# ==== K-Means ===================================================================================
# ================================================================================================

k = 3

centroids = init_centroids(k, data)

distance_table = cal_dist(centroids, data)

cluster_table = perf_clustering(k, distance_table)

newCentroids = update_centroids(centroids, cluster_table, data)

for i in range(1000):
    distance_table = cal_dist(newCentroids, data)

    cluster_table = perf_clustering(k, distance_table)

    newCentroids = update_centroids(newCentroids, cluster_table, data)

print(newCentroids)
