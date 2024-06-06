
from .visualizers import *

class PyodEvaluator:
    """Evaluator for Python Outlier Detection models
    """
    def __init__(self, config, models, data_test):
        """__init__

        :param config: A dictionary containing all the configuration variables for the experiment.
        :type config: dict
        :param models: Models to be evaluated.
        :type models: PyodModel
        :param data_test: Test dataset to evaluate the models.
        :type data_test: torch.utils.data.Dataset
        """
        self.config = config
        self.models = models
        self.data_test = data_test

    def evaluate(self):
        """Evaluate PyOD models. 

            - OutlierDetectionVisualizer: Visualization of test results.
        """
        if self.config['evaluation']['visualization']['activate']:
            visualizer = OutlierDetectionVisualizer(self.config, self.models, self.data_test, step_samples=self.config['arch']['step_samples_evaluation'])
            visualizer.visualize()
        