
import torch
import torchmetrics

def init_metrics(config):
    metrics_dict = {}

    metrics_list = config['evaluation']['metrics']
    for metric_name, metric_params in metrics_list.items():
        metrics_dict[metric_name] = getattr(torchmetrics, metric_name)(**metric_params)

    return metrics_dict

