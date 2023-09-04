#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from abc import ABC, abstractmethod
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchmetrics import ConfusionMatrix
from utils import *

class ClassificationVisualizerAbstract(ABC):
    """
    Evaluator generic class
    """
    def __init__(self, config, model, dataloader):
        self.config = config
        self.model = model
        self.test_loader = dataloader
        self.num_classes = self.config['data']['num_classes'] if self.config['task'] == 'Classification' else 1
        self.final_activation = self.config['implementation']['loss']['activation']
    
    def visualize(self, inference_outputs):
        """
        Visualize results
        """
        output = inference_outputs['outputs']
        labels = inference_outputs['labels']
        time = inference_outputs['time']
        event_names= inference_outputs['event_names']

        if self.config['task'] == 'Classification' and self.final_activation == 'linear':
            if self.num_classes == 1:
                output = [getattr(torch.nn.functional, 'sigmoid')(o) for o in output]
            else:
                output = [getattr(torch.nn.functional, 'softmax')(o, dim=1) for o in output]
        
        self.per_sample_operations(output, labels, time, event_names)
        self.global_operations()
    
    @abstractmethod
    def per_sample_operations(self):
        pass
    
    @abstractmethod
    def global_operations(self):
        pass