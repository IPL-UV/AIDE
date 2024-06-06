
import torch
import torchmetrics

def init_metrics(config):
    """Initialization of metric objects for evaluation purposes.

    :param config: A dictionary containing all the configuration variables for the experiment.
    :type config: dict
    :return: Dictionary with metrics from torchmetrics to be evaluated.
    :rtype: dict
    """
    metrics_dict = {}

    metrics_list = config['evaluation']['metrics']
    for metric_name, metric_params in metrics_list.items():
        metrics_dict[metric_name] = getattr(torchmetrics, metric_name)(**metric_params)

    return metrics_dict

