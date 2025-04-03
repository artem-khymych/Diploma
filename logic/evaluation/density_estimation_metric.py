import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KernelDensity
from scipy.stats import entropy, wasserstein_distance
from sklearn.utils import check_random_state

from project.logic.evaluation.metric_strategy import MetricStrategy


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

