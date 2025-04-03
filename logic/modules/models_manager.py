"""
Supervised Learning
    sklearn.linear_model
    sklearn.svm
    sklearn.tree
    sklearn.ensemble
    sklearn.neighbors
    sklearn.gaussian_process
    sklearn.cross_decomposition
    sklearn.naive_bayes
    sklearn.discriminant_analysis
"""
import inspect
from dataclasses import dataclass

import sklearn.linear_model as linear_model
import sklearn.ensemble as ensemble
import sklearn.svm as svm
import sklearn.neighbors as neighbors
import sklearn.tree as tree
import sklearn.naive_bayes as naive_bayes
import sklearn.discriminant_analysis as discriminant_analysis
import sklearn.cross_decomposition as cross_decomposition
import sklearn.gaussian_process as gaussian_process
from PyQt5.QtCore import pyqtSignal, QObject
from sklearn.base import is_regressor, is_classifier

from project.logic.modules import task_names

"""
Unsupervised Learning (Clustering, Dimensionality Reduction):
    sklearn.cluster
    sklearn.mixture
    sklearn.decomposition
    sklearn.manifold
    sklearn.covariance
"""
import sklearn.cluster as cluster
import sklearn.decomposition as decomposition
import sklearn.mixture as mixture
import sklearn.manifold as manifold
import sklearn.covariance as covariance

"""
Neural Networks:
    sklearn.neural_network 
"""
import sklearn.neural_network as neural_network

#TODO add statsmodels and more checks on models before sending dict
class ModelsManager(QObject):
    classification_modules = [
        linear_model, svm, tree, ensemble, neighbors,
        naive_bayes, discriminant_analysis
    ]

    regression_modules = [
        linear_model, svm, tree, ensemble, neighbors,
        gaussian_process, cross_decomposition
    ]

    clustering_modules = [cluster, mixture]

    dimension_reduction_modules = [decomposition, manifold]

    anomaly_detection_modules = [covariance, ensemble, neighbors]
    density_estimation_modules = [gaussian_process, mixture]

    neural_networks_modules = [neural_network]

    modules = {"Classification": classification_modules, "Regression": regression_modules,
               "Clustering": clustering_modules, "Dimensionality Reduction": dimension_reduction_modules,
               "Anomaly Detection": anomaly_detection_modules, "Density estimation": density_estimation_modules,
               "Scikit-learn MLP models": neural_networks_modules}

    models_dict_ready = pyqtSignal(dict)

    def create_models_dict(self, task):
        # create a dictionary of all classes in the listed modules
        model_dict = {}
        for module in self.modules[task]:
            for name, cls in inspect.getmembers(module, inspect.isclass):
                # Check if the class has both .fit and .predict methods
                if ((callable(getattr(cls, 'fit', None)) and callable(getattr(cls, 'predict', None)))
                        or callable(getattr(cls, 'transform', None)))\
                        or callable(getattr(cls, 'fit_transform', None)):

                    if not task == 'Classification' and not task == 'Regression':
                        model_dict[name] = cls

                    if task == task_names.REGRESSION and is_regressor(cls):
                        model_dict[name] = cls
                    if task == task_names.CLASSIFICATION and is_classifier(cls):
                        model_dict[name] = cls

        self.models_dict_ready.emit(model_dict)

    def get_model_by_name(self, name, model_dict):
        if name in model_dict:
            return model_dict[name]()
        else:
            raise ValueError(f"Model '{name}' not available in model_dict.")

    def get_model_params(self, model):
        model.get_params()
