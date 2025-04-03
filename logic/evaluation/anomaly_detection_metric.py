import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, f1_score,
    precision_score, recall_score, average_precision_score,
    confusion_matrix, accuracy_score, balanced_accuracy_score,
    cohen_kappa_score, matthews_corrcoef, fbeta_score
)
from scipy.stats import spearmanr, pearsonr

from project.logic.evaluation.metric_strategy import MetricStrategy


class AnomalyDetectionMetric(MetricStrategy):
    def evaluate(self, y_true, anomaly_scores, threshold=None, plot_curves=False, beta=1.0):
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

