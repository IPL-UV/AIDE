.. role:: raw-html(raw)
   :format: html

Metrics
===============

The metrics module provides a suite of functions to evaluate the performance of the machine learning algorithm. There metrics module is divided into two packages: `scikit-learn <https://scikit-learn.org/stable/api/sklearn.metrics.html>`_ for Outlier Detection and `TorchMetrics <https://lightning.ai/docs/torchmetrics/stable/>`_ for Detection and Impact Assessment. 
This means over 100+ evaluation metrics are implemented to evaluate the performance of the model.

Implementation
~~~~~~~~~~~~~~~~~~

The metrics module is implemented in the `evaluation` module, in which the metrics are defined. To use the metrics from the package, substitute :raw-html:`<font color="#008000">Metric_1</font>` in the code snippets below for the name of the class you want to use (e.g :raw-html:`<font color="#008000">roc_auc_score</font>` for AUROC from scikit-learn or :raw-html:`<font color="#008000">AUROC</font>` for TorchMetrics)


For OutlierDetection, specify also if the input to the metrics is probabilistic(y_scores) or the predictions(y_pred) with the parameter :raw-html:`<font color="#008000">probabilities</font>` (for more information, see scikit-learn documentation for each specific metric). Then, define as many metrics as you want as follows:

.. code-block:: yaml

    evaluation:
        metrics:
            Metric_1: {probabilities: 'True/False', parameter_1: 'value_parameter_1'}
            ...

For Detection and Impact Assessment, define as many metrics as you want as follows:

.. code-block:: yaml

    evaluation:
        metrics:
            Metric_1: {parameter_1: 'value_parameter_1'}
            ...