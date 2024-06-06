import numpy as np
from numpy import percentile
from sklearn.utils import column_or_1d
from sklearn.utils import check_consistent_length
import sklearn

def evaluate_print(clf_name, y, scores, config, threshold=0.5):
    """Utility function for evaluating and printing the results obtained given a PyOD model.
    Default metrics include ROC, Accuracy, Precision, Recall and Average Precision.

    :param clf_name: Name for the experiment.
    :type clf_name: str
    :param y: Target/ground-truth labels.
    :type y: np.array
    :param scores: Output scores provided by the model.
    :type scores: np.array
    :param threshold: Threshold to compute binary scores for evaluation purposes, defaults to 0.5
    :type threshold: float, optional
    """

    y = column_or_1d(y)
    scores = column_or_1d(scores)
    check_consistent_length(y, scores)

    y_pred = (scores > threshold).astype('int')

    metrics_string = clf_name + ': '
    metrics_list = config['evaluation']['metrics']
    for metric_name, metric_params in metrics_list.items():
        metric_params_dict = {param: metric_params[param] for param in set(list(metric_params.keys())) - set(['probabilities'])}
        if metric_params['probabilities']:
            metrics_string += (metric_name + ':')
            metrics_string += str(np.round(getattr(sklearn.metrics, metric_name)(y, scores, **metric_params_dict), decimals=4))
            metrics_string += ', '
        else:
            metrics_string += (metric_name + ':')
            metrics_string += str(np.round(getattr(sklearn.metrics, metric_name)(y, y_pred, **metric_params_dict), decimals=4))
            metrics_string += ', '
    
    print(metrics_string)