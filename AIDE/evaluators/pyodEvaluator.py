
from .visualizers import *

class PyodEvaluator:
    def __init__(self, config, models, data_test):
        self.config = config
        self.models = models
        self.data_test = data_test

    def evaluate(self):
        
        if self.config['evaluation']['visualization']['activate']:
            visualizer = OutlierDetectionVisualizer(self.config, self.models, self.data_test, step_samples=self.config['arch']['step_samples_evaluation'])
            visualizer.visualize()
        