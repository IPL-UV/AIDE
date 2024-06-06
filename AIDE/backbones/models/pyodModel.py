#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from importlib import import_module
from utils.metrics_pyod import evaluate_print

class PyodModel:
    """Template class for outlier detection models    
    
    :param config: configuration file
    :type config: dict
    """
    def __init__(self, config):
        """Constructor method
        """
        super().__init__()

        self.config = config
        self.name = self.config['arch']['type'].split('.')[1]

        self.__define_model()

    def forward(self, dataset, mode, step_samples=1):
        """Forward pass to get the model outputs 

        :param dataset: dataset variables
        :type dataset: numpy.ndarray
        :param mode: type of step: train, val or test
        :type mode: str
        :param step_samples: sampling interval to subset the dataset, defaults to 1
        :type step_samples: int, optional
        :return: results of the model
        :rtype: dict
        """
        x, labels = self.__adapt_variables(dataset)

        # Forward pass
        if mode == 'train':
            self.model.fit(x[::step_samples])
        output = self.model.predict_proba(x[::step_samples])[:,1]
        
        # Log metrics for evaluation
        self.__step_metrics(output, labels[::step_samples], mode=mode)
        
        return {'output': output, 'labels': labels}

    def get_model(self):
        """Calls for the handler of the model

        :return: Pyod model
        :rtype: pyod.models
        """
        return self.model

    def __define_model(self):
        """Defines the model from an external file (user defined)
        or the pyod library
        """
        if self.config['arch']['user_defined']:
            self.model = globals()[self.config['arch']['type']](config)
        else:        
            # IMPORT DYNAMICALLY PYOD MODEL
            module = import_module('pyod.models.'+self.config['arch']['type'].split('.')[0])
            self.model = getattr(module, self.config['arch']['type'].split('.')[1])
            self.model = self.model(**self.config['arch']['args'])
    
    def __adapt_variables(self, dataset):
        """Adapts the input variables and targets, applyies the masks

        :param dataset: dataset
        :type dataset: dict
        :return: input and target variables
        :rtype: numpy.ndarray
        """
        x = dataset['x']
        labels = dataset['labels']

        # Adapt_variables
        x = x.reshape(x.shape[0],-1).transpose()
        labels = labels.reshape(-1)
        if 'masks' in dataset.keys():
            masks = dataset['masks']
            masks = masks.reshape(masks.shape[0],-1).transpose()
            masks = np.all(masks,axis=1)
            x = x[masks]
            labels = labels[masks]

        return x, labels

    def __step_metrics(self, outputs, labels, mode):
        """Computes the metrics of each step

        :param outputs: ouptus of the model
        :type outputs: numpy.ndarray
        :param labels: target variables
        :type labels: numpy.ndarray
        :param mode: type of step: train, val or test
        :type mode: str
        """
        evaluate_print(self.name+'_'+mode, labels, outputs, self.config)
