import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, precision_recall_curve,
    average_precision_score, log_loss, cohen_kappa_score,
    matthews_corrcoef, balanced_accuracy_score, mean_squared_error, r2_score, classification_report
)
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score,
    homogeneity_score, completeness_score, v_measure_score,
    fowlkes_mallows_score
)
from sklearn.metrics.cluster import contingency_matrix
from scipy.spatial.distance import cdist

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    r2_score, median_absolute_error,
    explained_variance_score, max_error
)


# patern strategy
# TODO fill
class MetricStrategy:
    def evaluate(self, y_true, y_pred):
        raise NotImplementedError


class ClassificationMetric(MetricStrategy):
    def evaluate(self, y_true, y_pred, y_prob=None):
        """
        Ecaluates the classification experiment using different metrics.

        :param:
        -----------
        y_true : array-like
            True labels for classes.
        y_pred : array-like
            Predicted labels for classes.
        y_prob : array-like, optional
            Probability of predicted true clss, needed for
            ROC AUC, PR AUC and log loss. Optional

        :returns:
        -----------
        dict
            Dictionary with evaluated metrics.
        """
        metrics = {}

        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)

        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm

        # for binary classification
        if len(np.unique(y_true)) == 2:
            metrics['precision'] = precision_score(y_true, y_pred, average='binary')
            metrics['recall'] = recall_score(y_true, y_pred, average='binary')
            metrics['f1'] = f1_score(y_true, y_pred, average='binary')
            metrics['specificity'] = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0

            #
            if y_prob is not None:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
                metrics['pr_auc'] = average_precision_score(y_true, y_prob)
                metrics['log_loss'] = log_loss(y_true, y_prob)

        # for multi class classification
        else:
            metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro')
            metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro')
            metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted')

            metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro')
            metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro')
            metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted')

            metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
            metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro')
            metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')

            metrics['classification_report'] = classification_report(y_true, y_pred)
            if y_prob is not None and hasattr(y_prob, 'shape') and len(y_prob.shape) > 1:
                try:
                    metrics['log_loss'] = log_loss(y_true, y_prob)
                except:
                    pass

        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)

        return metrics


class RegressionMetric(MetricStrategy):
    def evaluate(self, y_true, y_pred):
        """
        Ecaluates the regression experiment using different metrics.

        :param:
        -----------
        y_true : array-like
            true values of target variable
        y_pred : array-like
            predicted values of target variable.

        :returns:
        -----------
        dict
            Dictionary with evaluated metrics.
        """
        metrics = {}

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Basic metrics
        metrics['mse'] = mean_squared_error(y_true, y_pred)  # Mean Squared Error
        metrics['rmse'] = np.sqrt(metrics['mse'])  # Root Mean Squared Error
        metrics['mae'] = mean_absolute_error(y_true, y_pred)  # Mean Absolute Error
        metrics['medae'] = median_absolute_error(y_true, y_pred)  # Median Absolute Error
        metrics['max_error'] = max_error(y_true, y_pred)  # Maximum Error

        # Relative error metrics
        metrics['mape'] = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100  # Mean Absolute Percentage Error
        metrics['smape'] = np.mean(
            2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-10)) * 100  # Symmetric MAPE

        # Quality metrics
        metrics['r2'] = r2_score(y_true, y_pred)  # R-squared (Coefficient of Determination)
        metrics['adjusted_r2'] = 1 - (1 - metrics['r2']) * (len(y_true) - 1) / (
            len(y_true) - y_pred.shape[1] if y_pred.ndim > 1 else len(y_true) - 1)  # Adjusted R-squared
        metrics['explained_variance'] = explained_variance_score(y_true, y_pred)  # Explained Variance

        errors = y_true - y_pred
        metrics['mean_error'] = np.mean(errors)  # Mean Error (ME)
        metrics['std_error'] = np.std(errors)  # Standard Deviation of Errors

        # Normalized error metrics
        y_true_var = np.var(y_true)
        if y_true_var > 0:
            metrics['nmse'] = metrics['mse'] / y_true_var  # Normalized Mean Squared Error
            metrics['nrmse'] = metrics['rmse'] / np.mean(y_true)  # Normalized Root Mean Squared Error
            metrics['rrse'] = np.sqrt(
                np.sum(np.square(errors)) / np.sum(np.square(y_true - np.mean(y_true))))  # Root Relative Squared Error
        else:
            metrics['nmse'] = float('inf')
            metrics['nrmse'] = float('inf')
            metrics['rrse'] = float('inf')

        # Theil's U statistic (version 2)
        sum_squared_pred = np.sum(np.square(y_pred))
        sum_squared_true = np.sum(np.square(y_true))
        if sum_squared_pred > 0 and sum_squared_true > 0:
            metrics['theil_u2'] = np.sqrt(np.sum(np.square(y_true - y_pred)) / sum_squared_true)
        else:
            metrics['theil_u2'] = float('inf')

        return metrics


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


import numpy as np
from sklearn.metrics import (
    mean_squared_error, explained_variance_score
)
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.manifold import trustworthiness


class DimReduction(MetricStrategy):
    def evaluate(self, X_original, X_reduced, X_reconstructed=None, n_neighbors=5,
                 precomputed_distances=None, precomputed_reduced_distances=None):
        """
        Обчислює різні метрики для оцінки якості зменшення розмірності.

        Параметри:
        -----------
        X_original : array-like
            Оригінальні дані високої розмірності.
        X_reduced : array-like
            Дані зниженої розмірності.
        X_reconstructed : array-like, optional
            Реконструйовані дані (з зниженої розмірності назад у вихідну).
            Потрібно для обчислення помилки реконструкції.
        n_neighbors : int, optional (default=5)
            Кількість сусідів для обчислення метрик збереження сусідства.
        precomputed_distances : array-like, optional
            Попередньо обчислена матриця відстаней для оригінальних даних.
        precomputed_reduced_distances : array-like, optional
            Попередньо обчислена матриця відстаней для даних зниженої розмірності.

        Повертає:
        -----------
        dict
            Словник з обчисленими метриками.
        """
        metrics = {}

        # Перевірка вхідних даних
        X_original = np.array(X_original)
        X_reduced = np.array(X_reduced)

        # Обчислення матриць відстаней, якщо вони не передані
        if precomputed_distances is None:
            dist_original = squareform(pdist(X_original))
        else:
            dist_original = precomputed_distances

        if precomputed_reduced_distances is None:
            dist_reduced = squareform(pdist(X_reduced))
        else:
            dist_reduced = precomputed_reduced_distances

        # 1. Кореляція Пірсона між відстанями
        # Перетворимо матриці відстаней у вектори (беремо верхні трикутні частини)
        n = dist_original.shape[0]
        triu_indices = np.triu_indices(n, k=1)
        dist_original_vec = dist_original[triu_indices]
        dist_reduced_vec = dist_reduced[triu_indices]

        # Обчислення кореляції
        pearson_corr, _ = pearsonr(dist_original_vec, dist_reduced_vec)
        metrics['pearson_correlation'] = pearson_corr

        # 2. Кореляція Спірмена між відстанями (ранговий коефіцієнт)
        spearman_corr, _ = spearmanr(dist_original_vec, dist_reduced_vec)
        metrics['spearman_correlation'] = spearman_corr

        # 3. Коефіцієнт надійності (Trustworthiness)
        # Вимірює, наскільки зберігаються локальні сусідства
        try:
            trust = trustworthiness(X_original, X_reduced, n_neighbors=n_neighbors)
            metrics['trustworthiness'] = trust
        except:
            metrics['trustworthiness'] = float('nan')

        # 4. Continuity (доповнення до Trustworthiness)
        # Continuity вимірює, чи нові сусіди в скороченому просторі
        # є справжніми сусідами в оригінальному просторі
        try:
            continuity = trustworthiness(X_reduced, X_original, n_neighbors=n_neighbors)
            metrics['continuity'] = continuity
        except:
            metrics['continuity'] = float('nan')

        # 5. Відсоток збереження дисперсії (для лінійних методів)
        # Якщо X_reconstructed не передано, можемо обчислити тільки
        # якщо розмірність X_reduced < X_original
        if X_original.shape[1] > X_reduced.shape[1]:
            # Дисперсія в зниженому просторі / дисперсія в оригінальному просторі
            var_original = np.sum(np.var(X_original, axis=0))
            var_reduced = np.sum(np.var(X_reduced, axis=0))
            metrics['variance_retention_ratio'] = var_reduced / var_original

        # 6. Метрики реконструкції (якщо надано реконструйовані дані)
        if X_reconstructed is not None:
            X_reconstructed = np.array(X_reconstructed)

            # Середньоквадратична помилка реконструкції
            mse = mean_squared_error(X_original, X_reconstructed)
            metrics['reconstruction_mse'] = mse
            metrics['reconstruction_rmse'] = np.sqrt(mse)

            # Відносна помилка реконструкції
            metrics['relative_reconstruction_error'] = np.sum((X_original - X_reconstructed) ** 2) / np.sum(
                X_original ** 2)

            # Коефіцієнт поясненої дисперсії
            metrics['explained_variance'] = explained_variance_score(X_original, X_reconstructed)

        # 7. K-найближчий сусід збереження (KNN Preservation)
        def knn_preservation(dist_orig, dist_red, k):
            """Обчислює відсоток k-найближчих сусідів, які зберігаються."""
            n = dist_orig.shape[0]
            knn_orig = np.argsort(dist_orig, axis=1)[:, 1:k + 1]  # Виключаємо саму точку
            knn_red = np.argsort(dist_red, axis=1)[:, 1:k + 1]

            preservation = 0
            for i in range(n):
                intersection = np.intersect1d(knn_orig[i], knn_red[i])
                preservation += len(intersection) / k

            return preservation / n

        # Обчислюємо збереження KNN для різних значень k
        for k in [5, 10, 20]:
            if n > k:  # Перевіряємо, що у нас достатньо даних
                knn_pres = knn_preservation(dist_original, dist_reduced, k)
                metrics[f'knn_preservation_{k}'] = knn_pres

        # 8. Відношення стресу (Stress Ratio)
        # Нормалізована сума квадратів різниць між відстанями
        sum_sq_dist_original = np.sum(dist_original ** 2)
        if sum_sq_dist_original > 0:
            stress = np.sqrt(np.sum((dist_original - dist_reduced) ** 2) / sum_sq_dist_original)
            metrics['stress_ratio'] = stress

        # 9. Локальна структура (обчислення для невеликих даних, оскільки це обчислювально інтенсивно)
        if n <= 1000:
            # Локальне масштабування відстаней Самона (LSDS - Local Scaling of Distances Score)
            # Високі значення означають краще збереження локальної структури
            def local_scaling(dist_mat, k=5):
                sigma = np.zeros(n)
                for i in range(n):
                    # k+1-й найближчий сусід (включаючи саму точку)
                    kth_distance = np.sort(dist_mat[i])[min(k + 1, n - 1)]
                    sigma[i] = kth_distance

                scaled_dist = np.zeros_like(dist_mat)
                for i in range(n):
                    for j in range(n):
                        scaled_dist[i, j] = dist_mat[i, j] / (sigma[i] * sigma[j])

                return scaled_dist

            try:
                scaled_original = local_scaling(dist_original, k=n_neighbors)
                scaled_reduced = local_scaling(dist_reduced, k=n_neighbors)

                # Кореляція між масштабованими відстанями
                scaled_original_vec = scaled_original[triu_indices]
                scaled_reduced_vec = scaled_reduced[triu_indices]

                lsds_corr, _ = pearsonr(scaled_original_vec, scaled_reduced_vec)
                metrics['local_scaling_correlation'] = lsds_corr
            except:
                metrics['local_scaling_correlation'] = float('nan')

        return metrics


from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, f1_score,
    precision_score, recall_score, average_precision_score,
    confusion_matrix, accuracy_score, balanced_accuracy_score,
    cohen_kappa_score, matthews_corrcoef, fbeta_score
)
from scipy.stats import spearmanr, pearsonr


class AnomalyDetectionMetric(MetricStrategy):
    def evaluate(y_true, anomaly_scores, threshold=None, plot_curves=False, beta=1.0):
        """
        Обчислює різні метрики для задачі виявлення аномалій.

        Параметри:
        -----------
        y_true : array-like
            Справжні мітки (1 для аномалій, 0 для нормальних спостережень).
        anomaly_scores : array-like
            Оцінки аномальності, обчислені алгоритмом (вищі значення означають
            більшу ймовірність аномалії).
        threshold : float, optional
            Поріг для перетворення оцінок аномальності в бінарні мітки.
            Якщо None, використовується поріг, що максимізує F1-score.
        plot_curves : bool, optional, default=False
            Якщо True, генерує графіки ROC і PR кривих.
        beta : float, optional, default=1.0
            Параметр для F-beta score. За замовчуванням = 1.0 (F1-score).

        Повертає:
        -----------
        dict
            Словник з обчисленими метриками.
        """
        # Перевірка вхідних даних
        y_true = np.array(y_true)
        anomaly_scores = np.array(anomaly_scores)

        # Переконаємося, що дані у правильному форматі (1 для аномалій, 0 для нормальних)
        if not np.array_equal(np.unique(y_true), np.array([0, 1])) and not np.array_equal(np.unique(y_true),
                                                                                          np.array([1, 0])):
            return {"error": "y_true має містити лише 0 (нормальні) та 1 (аномалії)"}

        metrics = {}

        # 1. Метрики, що не залежать від порогу
        try:
            # ROC AUC - Площа під ROC-кривою
            metrics['roc_auc'] = roc_auc_score(y_true, anomaly_scores)

            # PR AUC - Площа під кривою Precision-Recall
            metrics['pr_auc'] = average_precision_score(y_true, anomaly_scores)

            # Середній ранг аномалій
            rank_data = pd.DataFrame({
                'true': y_true,
                'score': anomaly_scores,
                'rank': pd.Series(anomaly_scores).rank(ascending=False)
            })
            anomaly_ranks = rank_data[rank_data['true'] == 1]['rank'].values
            metrics['average_rank_of_anomalies'] = np.mean(anomaly_ranks)
            metrics['median_rank_of_anomalies'] = np.median(anomaly_ranks)

            # Коефіцієнти кореляції
            spearman_corr, _ = spearmanr(y_true, anomaly_scores)
            pearson_corr, _ = pearsonr(y_true, anomaly_scores)
            metrics['spearman_correlation'] = spearman_corr
            metrics['pearson_correlation'] = pearson_corr
        except Exception as e:
            metrics['error_threshold_independent'] = str(e)

        # 2. Визначення оптимального порогу, якщо не вказано
        if threshold is None:
            precision, recall, thresholds = precision_recall_curve(y_true, anomaly_scores)
            # Обчислення F1 score для кожного порогу
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            # Визначення оптимального порогу (максимізація F1)
            optimal_idx = np.argmax(f1_scores)
            threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

        metrics['threshold'] = threshold
        y_pred = (anomaly_scores >= threshold).astype(int)

        # 3. Метрики на основі порогу
        try:
            # Confusion Matrix (Матриця помилок)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            metrics['true_positives'] = tp
            metrics['false_positives'] = fp
            metrics['true_negatives'] = tn
            metrics['false_negatives'] = fn

            # Базові метрики
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred)
            metrics['recall'] = recall_score(y_true, y_pred)  # Також відома як True Positive Rate (TPR)
            metrics['f1_score'] = f1_score(y_true, y_pred)

            # F-beta score
            metrics[f'f{beta}_score'] = fbeta_score(y_true, y_pred, beta=beta)

            # Додаткові метрики
            metrics['specificity'] = tn / (tn + fp)  # Також відома як True Negative Rate (TNR)
            metrics['false_positive_rate'] = fp / (fp + tn)  # 1 - specificity
            metrics['false_negative_rate'] = fn / (fn + tp)  # 1 - recall
            metrics['precision@n'] = precision_score(y_true, y_pred, zero_division=0)
            metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
            metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)

            # Метрики для незбалансованих даних
            n_anomalies = np.sum(y_true == 1)
            n_normal = np.sum(y_true == 0)
            metrics['prevalence'] = n_anomalies / (n_anomalies + n_normal)

            # Геометричне середнє precision і recall
            if metrics['precision'] > 0 and metrics['recall'] > 0:
                metrics['g_mean'] = np.sqrt(metrics['precision'] * metrics['recall'])
            else:
                metrics['g_mean'] = 0.0

        except Exception as e:
            metrics['error_threshold_dependent'] = str(e)

        # 4. Розрахунок додаткових метрик для ранжування
        try:
            # Відсоток виявлених аномалій у верхніх N%
            percentiles = [1, 5, 10, 20]
            for p in percentiles:
                n_top = int(np.ceil(len(y_true) * p / 100))
                top_indices = np.argsort(anomaly_scores)[-n_top:]
                top_precision = np.sum(y_true[top_indices]) / n_top
                metrics[f'precision_top_{p}%'] = top_precision

            # Детекція при різних рівнях виявлення аномалій
            detection_rates = [0.5, 0.8, 0.9, 0.95, 0.99]
            for rate in detection_rates:
                n_anomalies = np.sum(y_true == 1)
                n_to_detect = int(np.ceil(n_anomalies * rate))

                if n_to_detect > 0:
                    # Сортуємо за оцінками аномальності (у спадному порядку)
                    sorted_indices = np.argsort(anomaly_scores)[::-1]
                    sorted_true = y_true[sorted_indices]

                    # Знаходимо позицію, де ми виявили вказаний відсоток аномалій
                    cum_true_pos = np.cumsum(sorted_true)
                    idx = np.argmax(cum_true_pos >= n_to_detect)

                    # Порахуємо скільки спостережень потрібно перевірити
                    total_to_check = idx + 1
                    metrics[f'detection_rate_{int(rate * 100)}%'] = total_to_check / len(y_true)
        except Exception as e:
            metrics['error_ranking_metrics'] = str(e)

        return metrics


import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KernelDensity
from scipy.stats import entropy, wasserstein_distance
from sklearn.utils import check_random_state


class DensityEstimationMetric(MetricStrategy):

    def evaluate(self, X, density_estimator, X_test=None, reference_density=None,
                 grid_points=1000, range_min=None, range_max=None,
                 n_folds=5, random_state=None, plot=False):
        """
        Обчислює різні метрики для оцінки алгоритмів Density Estimation.

        Параметри:
        -----------
        X : array-like
            Навчальні дані для оцінки густини.
        density_estimator : object
            Об'єкт sklearn, який реалізує метод score для обчислення log-правдоподібності
            та метод predict_proba або score_samples для обчислення густини.
        X_test : array-like, optional
            Тестові дані. Якщо не вказано, використовується крос-валідація на X.
        reference_density : callable, optional
            Еталонна функція густини для порівняння (якщо відома).
            Якщо None, ця метрика не обчислюється.
        grid_points : int, optional
            Кількість точок для дискретизації простору для обчислення деяких метрик.
        range_min : float or array-like, optional
            Мінімальне значення діапазону для кожної змінної. Якщо None,
            використовується мінімум даних.
        range_max : float or array-like, optional
            Максимальне значення діапазону для кожної змінної. Якщо None,
            використовується максимум даних.
        n_folds : int, optional
            Кількість фолдів для крос-валідації.
        random_state : int or RandomState, optional
            Випадковий стан для відтворюваності.
        plot : bool, optional
            Чи створювати графіки порівняння (для одновимірних даних).

        Повертає:
        -----------
        dict
            Словник з обчисленими метриками.
        """
        metrics = {}

        # Перевірка вхідних даних
        X = np.array(X)
        n_samples, n_features = X.shape
        rng = check_random_state(random_state)

        # Визначення діапазону для оцінки густини
        if range_min is None:
            range_min = np.min(X, axis=0) - 0.1 * np.std(X, axis=0)
        if range_max is None:
            range_max = np.max(X, axis=0) + 0.1 * np.std(X, axis=0)

        if np.isscalar(range_min):
            range_min = np.array([range_min] * n_features)
        if np.isscalar(range_max):
            range_max = np.array([range_max] * n_features)

        # 1. Log-правдоподібність (повертається як середнє значення на один зразок)
        if hasattr(density_estimator, 'score'):
            if X_test is not None:
                log_likelihood = density_estimator.score(X_test)
                metrics['log_likelihood'] = log_likelihood
            else:
                # Використовуємо крос-валідацію для оцінки log-правдоподібності
                kf = KFold(n_splits=n_folds, shuffle=True, random_state=rng)
                log_likelihoods = []

                for train_idx, test_idx in kf.split(X):
                    X_train, X_val = X[train_idx], X[test_idx]

                    # Створюємо копію оцінювача для навчання
                    estimator_copy = self.clone_estimator(density_estimator)
                    try:
                        estimator_copy.fit(X_train)
                        log_likelihoods.append(estimator_copy.score(X_val))
                    except:
                        # Якщо клонування не працює, видаємо повідомлення
                        metrics[
                            'log_likelihood_cv'] = "Помилка при крос-валідації. Використовуйте вже навчений оцінювач і передавайте X_test"
                        break

                if log_likelihoods:
                    metrics['log_likelihood_cv'] = np.mean(log_likelihoods)

        # 2. Обчислення інтегральних метрик (для одновимірних даних)
        if n_features == 1 and reference_density is not None:
            # Створюємо сітку для оцінки
            x_grid = np.linspace(range_min[0], range_max[0], grid_points)
            x_grid_reshaped = x_grid.reshape(-1, 1)

            # Отримуємо оцінки густини
            if hasattr(density_estimator, 'score_samples'):
                estimated_log_density = density_estimator.score_samples(x_grid_reshaped)
                estimated_density = np.exp(estimated_log_density)
            elif hasattr(density_estimator, 'predict_proba'):
                estimated_density = density_estimator.predict_proba(x_grid_reshaped)
            else:
                raise ValueError("Оцінювач густини повинен мати score_samples або predict_proba")

            # Нормалізація для забезпечення інтегралу в 1
            estimated_density = estimated_density / (np.sum(estimated_density) * (x_grid[1] - x_grid[0]))

            # Обчислення еталонної густини
            reference_values = np.array([reference_density(x) for x in x_grid])

            # Нормалізація для забезпечення інтегралу в 1
            reference_values = reference_values / (np.sum(reference_values) * (x_grid[1] - x_grid[0]))

            # 2.1 Відстань Кульбака-Лейблера (KL-дивергенція)
            # Уникаємо ділення на нуль
            eps = 1e-10
            reference_values_safe = np.maximum(reference_values, eps)
            estimated_density_safe = np.maximum(estimated_density, eps)

            kl_div = entropy(reference_values_safe, estimated_density_safe)
            metrics['kl_divergence'] = kl_div

            # 2.2 Відстань Вассерштейна (Earth Mover's Distance)
            wd = wasserstein_distance(x_grid, x_grid, reference_values, estimated_density)
            metrics['wasserstein_distance'] = wd

            # 2.3 Середньоквадратична помилка
            mse = np.mean((reference_values - estimated_density) ** 2)
            metrics['density_mse'] = mse

            # 2.4 Максимальна абсолютна різниця
            max_diff = np.max(np.abs(reference_values - estimated_density))
            metrics['max_absolute_difference'] = max_diff

            # 2.5 Середня абсолютна різниця
            mean_abs_diff = np.mean(np.abs(reference_values - estimated_density))
            metrics['mean_absolute_difference'] = mean_abs_diff

            # 2.6 Перевірка нормалізації (інтеграл повинен бути близько 1)
            integral_approx = np.sum(estimated_density) * (x_grid[1] - x_grid[0])
            metrics['integral_value'] = integral_approx

        # 3. Оцінка щільності в тестових точках (для багатовимірних даних)
        if X_test is not None:
            X_test = np.array(X_test)

            if hasattr(density_estimator, 'score_samples'):
                log_densities = density_estimator.score_samples(X_test)
                densities = np.exp(log_densities)

                metrics['mean_test_density'] = np.mean(densities)
                metrics['min_test_density'] = np.min(densities)
                metrics['max_test_density'] = np.max(densities)

                # Відсоток точок з низькою густиною (потенційні викиди)
                quantile_5 = np.quantile(densities, 0.05)
                metrics['low_density_ratio'] = np.mean(densities < quantile_5)

        # 4. Метрики для багатовимірних даних без еталонної густини
        # 4.1 Інформаційний критерій Акаіке (AIC)
        if hasattr(density_estimator, 'score') and hasattr(density_estimator, 'n_parameters_'):
            log_likelihood = density_estimator.score(X) * n_samples  # Повна log-правдоподібність
            k = density_estimator.n_parameters_ if hasattr(density_estimator, 'n_parameters_') else 1
            aic = 2 * k - 2 * log_likelihood
            metrics['aic'] = aic

        # 4.2 Байєсівський інформаційний критерій (BIC)
        if hasattr(density_estimator, 'score') and hasattr(density_estimator, 'n_parameters_'):
            log_likelihood = density_estimator.score(X) * n_samples  # Повна log-правдоподібність
            k = density_estimator.n_parameters_ if hasattr(density_estimator, 'n_parameters_') else 1
            bic = k * np.log(n_samples) - 2 * log_likelihood
            metrics['bic'] = bic

        return metrics

    def clone_estimator(self, estimator):
        """Спроба клонувати оцінювач або повернути None, якщо не вдалося."""
        try:
            from sklearn.base import clone
            return clone(estimator)
        except:
            # Якщо не вдалося клонувати за допомогою sklearn, спробуємо альтернативний підхід
            try:
                import copy
                return copy.deepcopy(estimator)
            except:
                # Якщо обидва підходи не працюють, це означає, що оцінювач не можна клонувати
                return None


import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)
from scipy import stats
from statsmodels.tsa.stattools import acf, pacf, adfuller, kpss
import warnings


class TimeSeriesMetric(MetricStrategy):
    def evaluate(y_true, y_pred, y_naive=None, residuals=None, alpha=0.05,
                 seasonality=None, freq=None, return_diagnostics=False):
        """
        Обчислює метрики для оцінки якості моделей часових рядів.

        Параметри:
        -----------
        y_true : array-like
            Справжні значення часового ряду.
        y_pred : array-like
            Прогнозовані значення часового ряду.
        y_naive : array-like, optional
            Прогноз наївної моделі (наприклад, просте зміщення на один крок).
            Якщо не вказано, використовується просте значення з попереднього кроку.
        residuals : array-like, optional
            Залишки моделі. Якщо не вказано, обчислюються як y_true - y_pred.
        alpha : float, optional (default=0.05)
            Рівень значущості для інтервалів довіри та статистичних тестів.
        seasonality : int, optional
            Період сезонності (наприклад, 12 для місячних даних з річною сезонністю).
        freq : str, optional
            Частота часового ряду для масштабування деяких метрик (наприклад, 'D' для щоденних даних).
        return_diagnostics : bool, optional (default=False)
            Якщо True, повертає додаткові діагностичні графіки та дані для залишків.

        Повертає:
        -----------
        dict
            Словник з обчисленими метриками.
        """
        metrics = {}

        # Перевірка вхідних даних
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        if len(y_true) != len(y_pred):
            raise ValueError("y_true і y_pred повинні мати однакову довжину")

        # 1. Базові метрики точності прогнозування

        # Середня абсолютна похибка
        metrics['mae'] = mean_absolute_error(y_true, y_pred)

        # Середньоквадратична похибка
        metrics['mse'] = mean_squared_error(y_true, y_pred)

        # Корінь із середньоквадратичної похибки
        metrics['rmse'] = np.sqrt(metrics['mse'])

        # Середня абсолютна процентна похибка
        # Уникаємо ділення на нуль
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred) * 100
            except:
                nonzero_idx = y_true != 0
                if np.any(nonzero_idx):
                    metrics['mape'] = np.mean(
                        np.abs((y_true[nonzero_idx] - y_pred[nonzero_idx]) / y_true[nonzero_idx])) * 100
                else:
                    metrics['mape'] = np.nan

        # Симетрична MAPE (sMAPE)
        denominator = np.abs(y_true) + np.abs(y_pred)
        nonzero_idx = denominator != 0
        if np.any(nonzero_idx):
            metrics['smape'] = np.mean(
                2 * np.abs(y_true[nonzero_idx] - y_pred[nonzero_idx]) / denominator[nonzero_idx]) * 100
        else:
            metrics['smape'] = np.nan

        # Середня абсолютна масштабована похибка (MASE)
        # Потрібен базовий наївний прогноз для масштабування
        if y_naive is None and len(y_true) > 1:
            # Використовуємо просту модель: значення з попереднього кроку
            y_naive = np.concatenate([[y_true[0]], y_true[:-1]])

        if y_naive is not None:
            naive_errors = np.abs(y_true - y_naive)
            if np.sum(naive_errors) > 0:
                metrics['mase'] = np.mean(np.abs(y_true - y_pred)) / np.mean(naive_errors)
            else:
                metrics['mase'] = np.nan

        # Коефіцієнт детермінації R²
        metrics['r2'] = r2_score(y_true, y_pred)

        # 2. Аналіз залишків

        if residuals is None:
            residuals = y_true - y_pred

        # Середнє значення залишків (має бути близьким до нуля)
        metrics['residual_mean'] = np.mean(residuals)

        # Стандартне відхилення залишків
        metrics['residual_std'] = np.std(residuals)

        # Тест Дікі-Фуллера для стаціонарності залишків
        try:
            adf_result = adfuller(residuals)
            metrics['adf_statistic'] = adf_result[0]
            metrics['adf_pvalue'] = adf_result[1]
            metrics['residuals_stationary'] = adf_result[1] < alpha
        except:
            metrics['adf_statistic'] = np.nan
            metrics['adf_pvalue'] = np.nan
            metrics['residuals_stationary'] = None

        # Тест Квятковського-Філліпса-Шмідта-Шина (KPSS) для стаціонарності
        try:
            kpss_result = kpss(residuals)
            metrics['kpss_statistic'] = kpss_result[0]
            metrics['kpss_pvalue'] = kpss_result[1]
            metrics['residuals_trend_stationary'] = kpss_result[1] > alpha
        except:
            metrics['kpss_statistic'] = np.nan
            metrics['kpss_pvalue'] = np.nan
            metrics['residuals_trend_stationary'] = None

        # Тест Бокса-Пірса для автокореляції
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_result = acorr_ljungbox(residuals, lags=min(10, len(residuals) // 5))
            metrics['ljung_box_statistic'] = lb_result.iloc[-1, 0]
            metrics['ljung_box_pvalue'] = lb_result.iloc[-1, 1]
            metrics['residuals_independent'] = lb_result.iloc[-1, 1] > alpha
        except:
            metrics['ljung_box_statistic'] = np.nan
            metrics['ljung_box_pvalue'] = np.nan
            metrics['residuals_independent'] = None

        # Тест Харке-Бера для нормальності
        try:
            jb_stat, jb_pvalue = stats.jarque_bera(residuals)
            metrics['jarque_bera_statistic'] = jb_stat
            metrics['jarque_bera_pvalue'] = jb_pvalue
            metrics['residuals_normal'] = jb_pvalue > alpha
        except:
            metrics['jarque_bera_statistic'] = np.nan
            metrics['jarque_bera_pvalue'] = np.nan
            metrics['residuals_normal'] = None

        # Автокореляція залишків
        try:
            residual_acf = acf(residuals, nlags=min(20, len(residuals) // 4), fft=True)
            # Кількість значущих лагів автокореляції
            confidence_interval = stats.norm.ppf(1 - alpha / 2) / np.sqrt(len(residuals))
            significant_lags = np.sum(np.abs(residual_acf[1:]) > confidence_interval)
            metrics['significant_acf_lags'] = significant_lags

            # Часткова автокореляція залишків
            residual_pacf = pacf(residuals, nlags=min(20, len(residuals) // 4))
            # Кількість значущих лагів часткової автокореляції
            significant_pacf_lags = np.sum(np.abs(residual_pacf[1:]) > confidence_interval)
            metrics['significant_pacf_lags'] = significant_pacf_lags
        except:
            metrics['significant_acf_lags'] = np.nan
            metrics['significant_pacf_lags'] = np.nan

        # 3. Спеціалізовані метрики для часових рядів

        # Тіл U статистика (Theil's U) - порівнює з наївним прогнозом
        if y_naive is not None:
            naive_mse = mean_squared_error(y_true, y_naive)
            if naive_mse > 0:
                metrics['theils_u'] = np.sqrt(metrics['mse'] / naive_mse)
            else:
                metrics['theils_u'] = np.nan

        # Оцінка точності напрямку (Direction Accuracy)
        if len(y_true) > 1:
            actual_direction = np.sign(y_true[1:] - y_true[:-1])
            pred_direction = np.sign(y_pred[1:] - y_pred[:-1])
            metrics['direction_accuracy'] = np.mean(actual_direction == pred_direction) * 100

        # Пікова відносна помилка (PRE)
        peak_idx = np.argmax(np.abs(y_true))
        metrics['peak_error'] = np.abs(y_true[peak_idx] - y_pred[peak_idx])
        if y_true[peak_idx] != 0:
            metrics['peak_relative_error'] = metrics['peak_error'] / np.abs(y_true[peak_idx]) * 100
        else:
            metrics['peak_relative_error'] = np.nan

        # 4. Сезонні метрики (якщо вказано сезонність)
        if seasonality is not None and len(y_true) >= 2 * seasonality:
            # Розділити дані на сезонні компоненти
            n_seasons = len(y_true) // seasonality
            seasonal_errors = np.zeros(seasonality)

            for i in range(seasonality):
                season_indices = [i + s * seasonality for s in range(n_seasons) if i + s * seasonality < len(y_true)]
                if season_indices:
                    seasonal_errors[i] = np.mean(np.abs(y_true[season_indices] - y_pred[season_indices]))

            metrics['max_seasonal_error'] = np.max(seasonal_errors)
            metrics['min_seasonal_error'] = np.min(seasonal_errors)
            metrics['seasonal_error_std'] = np.std(seasonal_errors)
            metrics['seasonal_error_ratio'] = metrics['max_seasonal_error'] / metrics['min_seasonal_error'] if metrics[
                                                                                                                   'min_seasonal_error'] > 0 else np.nan

        # 5. Додаткові діагностичні дані
        if return_diagnostics:
            diagnostics = {}

            # ACF і PACF залишків
            try:
                diagnostics['residual_acf'] = acf(residuals, nlags=min(40, len(residuals) // 2), fft=True)
                diagnostics['residual_pacf'] = pacf(residuals, nlags=min(40, len(residuals) // 2))
            except:
                diagnostics['residual_acf'] = None
                diagnostics['residual_pacf'] = None

            # QQ-plot дані
            try:
                from scipy.stats import probplot
                qq_data = probplot(residuals, dist='norm')
                diagnostics['qq_plot_data'] = qq_data
            except:
                diagnostics['qq_plot_data'] = None

            # Розподіл помилок за величиною
            error_bins = np.histogram(np.abs(residuals), bins=10)
            diagnostics['error_distribution'] = error_bins

            # Повертаємо обидва результати - метрики та діагностику
            return metrics, diagnostics

        return metrics


class NNMetric(MetricStrategy):
    def evaluate(self):
        pass
