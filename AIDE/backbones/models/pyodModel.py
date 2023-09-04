#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from importlib import import_module
from utils.metrics_pyod import evaluate_print

class PyodModel:
    """
    Template class for outlier detection methods
    """    
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.name = self.config['arch']['type'].split('.')[1]

        self.__define_model()

    def forward(self, dataset, mode, step_samples=1):
        x, labels = self.__adapt_variables(dataset)

        # Forward pass
        if mode == 'train':
            self.model.fit(x[::step_samples])
        output = self.model.predict_proba(x[::step_samples])[:,1]
        
        # Log metrics for evaluation
        self.__step_metrics(output, labels[::step_samples], mode=mode)
        
        return {'output': output, 'labels': labels}

    def get_model(self):
        return self.model

    def __define_model(self):
        if self.config['arch']['user_defined']:
            self.model = globals()[self.config['arch']['type']](config)
        else:        
            # IMPORT DYNAMICALLY PYOD MODEL
            module = import_module('pyod.models.'+self.config['arch']['type'].split('.')[0])
            self.model = getattr(module, self.config['arch']['type'].split('.')[1])
            self.model = self.model(**self.config['arch']['args'])
    
    def __adapt_variables(self, dataset):
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
        """
        Compute and print evaluation metrics
        """
        evaluate_print(self.name+'_'+mode, labels, outputs)
