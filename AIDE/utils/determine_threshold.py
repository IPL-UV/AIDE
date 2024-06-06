import numpy as np
import torch
from numpy import percentile
from sklearn.utils import column_or_1d
import torchmetrics

def determine_threshold(y_pred, y_true, metric_name = "f1_score", metric_params = {}, num = 100, lower_is_better = False):
    """Find optimal threshold based on defined metric.

    :param y_pred: Prediction/output variables by your model.
    :type y_pred: Array
    :param y_true: Target/ground-truth labels.
    :type y_true: Array
    :param metric_name: Torchmetrics for threshold optimization, defaults to "f1_score".
    :type metric_name: Str
    :param metric_params: Parameters for Torchmetrics, defaults to {}.
    :type metric_params: Dict
    :param num: Number of candidates for threshold optimization, defaults to 100.
    :type num: Int
    :param lower_is_better: Indicate whether or not lower metric value is better, defaults to False.
    :type lower_is_better: Float

    :return: Dictionary with metrics from torchmetrics to be evaluated.
    :rtype: Dict
    """
    y_pred = torch.Tensor(y_pred.flatten())
    y_true = torch.Tensor(y_true.flatten())
    assert len(y_pred) == len(y_true), 'len(y_pred) != len(y_true)'
    
    try:
        metric = getattr(torchmetrics, metric_name)(**metric_params)
        metric = metric.to(y_pred.device)
        
        print('Optimizing threshold')
        candidates = torch.unique(y_pred).numpy()
        candidates = candidates[::np.round(len(candidates)/num).astype('int')]

        optimal_threshold_value = [-1., None]
        for i, threshold in enumerate(candidates):
            value = metric((y_pred >= threshold).type(y_true.dtype), y_true)
            
            if lower_is_better and ((optimal_threshold_value[1] is None) or optimal_threshold_value[1] > value):
                optimal_threshold_value = [threshold, value]
            
            elif not lower_is_better and ((optimal_threshold_value[1] is None) or optimal_threshold_value[1] < value):
                optimal_threshold_value = [threshold, value]
            
            metric.reset()
    except:
        print(metric_name+ ' not found in torchmetrics for threshold optimization.')
        optimal_threshold_value = [0.5]
    
    return optimal_threshold_value[0]
