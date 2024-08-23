# Clustering metric notes

## Davies-Bouldin index

$X_j$ is a point, $C_i$ is a cluster. $S_i$ is the average distance from the point to the centroid in cluster $C_i$. $M_{i,j}$ is distance between clusters $C_i$ and $C_j$.

$R_{i,j}=\frac{S_i+S_j}{M_{i,j}}$.  This is bigger if clusters are highly variant or the distances between clusters is small.

$D_i$ is the largest $R_{i,j}$ over all other clusters $j$.  This is bigger if any one cluster is too close to $C_i$ or is highly variant.

The Davies-Bouldin index is the average $D_i$ over all clusters.

### DB is larger (worse) if:

- Clusters are spread out and/or close together
- Any *one* cluster is highly variant or poorly formed

## Calinski-Harabasz Index

$CH=\frac{BCSS/(k-1)}{WCSS/(n-k)}$

BCSS is the weighted average of the distance from the cluster centroids to the overall data centroid

WCSS is the average distance from a point to its cluster centroid.

$k$ is the number of clusters.

### CH is smaller (worse) if:

- cluster centroids (especially large clusters) are close to the overall centroid
- data points are on average far away from their centroids
- the number of clusters is large

## Silhouette

$a_i$ is the average distance from data point $i$ to the other points in its assigned cluster.

To calculate $b_i$ for data point $i$, calculate the average distance to each point in each other cluster - the minimum such average distance is $b_i$. This is the average distance to the data in the next-best cluster for data point $i$.

The silhouette value of data point $i$ is $s_i=\frac{b_i-a_i}{\max{a_i,b_i}}$. If $b_i>a_i$ (which I would think it should be usually?), this is the same as $1-\frac{a_i}{b_i}$.

The silhouette score of all the data points is the mean of the silhouette values.

### Silhouette is smaller (worse) if:

- the next-best clusters are almost as close as the current cluster

You can also look at silhouette scores of individual clusters to identify good and bad clusters
