import numpy as np

from project.logic.evaluation.metric_strategy import MetricStrategy
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score,
    homogeneity_score, completeness_score, v_measure_score,
    fowlkes_mallows_score
)
from sklearn.metrics.cluster import contingency_matrix
from scipy.spatial.distance import cdist


class ClusteringMetric(MetricStrategy):
    def evaluate(self, X, labels, true_labels=None):
        """
        Ecaluates the clustering experiment using different metrics.

        :param:
        -----------
        X : array-like
            Matrix of data points.
        labels : array-like
            Labels corresponding to clusters.
        true_labels : array-like, optional
            True labels corresponding to clusters. If we have them. Optional

        :returns:
        -----------
        dict
            Dictionary with evaluated metrics.
        """
        metrics = {}

        X = np.array(X)
        labels = np.array(labels)
        n_clusters = len(np.unique(labels[labels != -1]))

        if n_clusters < 2:
            return {"error": "Потрібно мати щонайменше 2 кластери для обчислення метрик"}

        # Inner metrics - dont need y_true labels
        try:
            metrics['silhouette_score'] = silhouette_score(X, labels)
        except:
            metrics['silhouette_score'] = float('nan')

        try:
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(X, labels)
        except:
            metrics['calinski_harabasz_score'] = float('nan')

        try:
            metrics['davies_bouldin_score'] = davies_bouldin_score(X, labels)
        except:
            metrics['davies_bouldin_score'] = float('nan')

        # Inertia
        cluster_centers = np.array([X[labels == i].mean(axis=0) for i in range(n_clusters)])
        inertia = 0
        for i in range(n_clusters):
            if np.sum(labels == i) > 0:
                cluster_points = X[labels == i]
                inertia += np.sum(np.square(cdist(cluster_points, [cluster_centers[i]], 'euclidean')))
        metrics['inertia'] = inertia

        # Average distance between clusters - euclidean metric
        within_distances = []
        for i in range(n_clusters):
            if np.sum(labels == i) > 1:
                cluster_points = X[labels == i]
                distances = cdist(cluster_points, cluster_points, 'euclidean')
                within_distances.append(np.sum(distances) / (2 * len(cluster_points)))
        if within_distances:
            metrics['avg_within_cluster_distance'] = np.mean(within_distances)
        else:
            metrics['avg_within_cluster_distance'] = float('nan')

        # Average distance between clusters centers - euclidean metric
        if n_clusters > 1:
            between_distances = cdist(cluster_centers, cluster_centers, 'euclidean')
            metrics['avg_between_cluster_distance'] = np.sum(between_distances) / (n_clusters * (n_clusters - 1))
        else:
            metrics['avg_between_cluster_distance'] = float('nan')

        # Outer metrics - need y_true labels
        if true_labels is not None:
            true_labels = np.array(true_labels)

            try:
                # Adjusted Rand Index
                metrics['adjusted_rand_score'] = adjusted_rand_score(true_labels, labels)
            except:
                metrics['adjusted_rand_score'] = float('nan')

            try:
                # Normalized Mutual Information
                metrics['normalized_mutual_info_score'] = normalized_mutual_info_score(true_labels, labels)
            except:
                metrics['normalized_mutual_info_score'] = float('nan')

            try:
                # Adjusted Mutual Information
                metrics['adjusted_mutual_info_score'] = adjusted_mutual_info_score(true_labels, labels)
            except:
                metrics['adjusted_mutual_info_score'] = float('nan')

            try:
                # Homogeneity, Completeness and V-measure
                metrics['homogeneity_score'] = homogeneity_score(true_labels, labels)
                metrics['completeness_score'] = completeness_score(true_labels, labels)
                metrics['v_measure_score'] = v_measure_score(true_labels, labels)
            except:
                metrics['homogeneity_score'] = float('nan')
                metrics['completeness_score'] = float('nan')
                metrics['v_measure_score'] = float('nan')

            try:
                # Fowlkes-Mallows score
                metrics['fowlkes_mallows_score'] = fowlkes_mallows_score(true_labels, labels)
            except:
                metrics['fowlkes_mallows_score'] = float('nan')

            try:
                cm = contingency_matrix(true_labels, labels)
                metrics['contingency_matrix'] = cm

                purity = 0
                for j in range(cm.shape[1]):
                    purity += np.max(cm[:, j])
                metrics['purity'] = purity / np.sum(cm)
            except:
                metrics['purity'] = float('nan')

        return metrics

