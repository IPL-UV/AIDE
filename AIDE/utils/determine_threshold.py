import numpy as np
import torch
from numpy import percentile
from sklearn.utils import column_or_1d
import torchmetrics

def determine_threshold(y_pred, y_true, metric_name = "f1_score", metric_params = {}, num = 100, lower_is_better = False):
    """
    Find optimal threshold based on defined metric
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
