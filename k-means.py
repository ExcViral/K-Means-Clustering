# import the necessary libraries

import math
import pandas as pd
import numpy as np

# import the dataset

df = pd.read_csv("iris.csv")

# print(df)

# fetch all features in different sets

id = df['id'].values
f1 = df['sepallengthcm'].values
f2 = df['sepalwidthcm'].values
f3 = df['petallengthcm'].values
f4 = df['petalwidthcm'].values

data = np.array(list(zip(f1, f2, f3, f4)))


# print(len(data[0]))

# ======================================================================

# function to calculate euclidean distance between two rows

def euc_dist(a, b):
    sum = 0
    for i, j in zip(a, b):
        a = (i - j) * (i - j)
        sum = sum + a
    return math.sqrt(sum)


# ======================================================================

# Actual implementation of k-means

# Step 1: Choose the number of clusters 'k'

k = 3

# Step 2: Initialize 'k' numbers of centroids randomly

c = np.random.rand(k, len(data[0]))
# print(c)

# Step 3: Calculate distances of all data members w.r.t centroids

c_dist = []

for i in c:
    temp = []
    for j in data:
        temp.append(euc_dist(i, j))
    c_dist.append(temp)
    temp = []

# Step 4: Clustering: for a data point d, cluster it with the centroid of minimum distance c

# create empty cluster list of size k
clusters = []
for i in range(k):
    clusters.append([])

# start clustering with reference to distance table generated in previous step
for i in range(len(c_dist[0])):
    d = []
    for j in range(len(c_dist)):
        d.append(c_dist[j][i])
    clusters[d.index(min(d))].append(i)

# Step 5: Update centroid locations

for i in range(len(c)):
    if (len(clusters[i]) > 0):
        temp = []
        for j in clusters[i]:
            temp.append(list(data[j]))

        sum = [0] * len(c[i])
        for l in temp:
            sum = [(a + b) for a, b in zip(sum, l)]

        c[i] = [p / len(temp) for p in sum]

print("updated centroids: ", c)
