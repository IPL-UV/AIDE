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

class VisualizerAbstract(ABC):
    """
    Evaluator generic class
    """
    def __init__(self, config, model, dataloader):
        """Initialization of the ClassificationVisualizer's parameters

        :param config: The configuration file
        :type config: dict
        :param model: PyTorch trained model for evaluation
        :type model: class: 'torch.nn.Module'
        :param dataloader: PyTorch data iterator
        :type dataloader: class: 'torch.utils.data.DataLoader'
        """

        self.config = config
        self.model = model
        self.test_loader = dataloader
        self.num_classes = self.config['data']['num_classes'] if self.config['task'] == 'Classification' else 1
        self.final_activation = self.config['implementation']['loss']['activation']
    
    def visualize(self, inference_outputs):
        """Visualization of results

        :param inference_outputs: Dictionary containing the variables to perform the visualization over 
        :type inference_outputs: dict
        """
        output = inference_outputs['outputs']
        labels = inference_outputs['labels']
        time = inference_outputs['time']
        event_names= inference_outputs['event_names']
        masks= inference_outputs['masks'] if self.config['implementation']['loss']['masked'] else [None]*len(event_names)
        

        if self.final_activation['type'] in ['DeepGP']:
            uncertainties = {'outputs_upper': inference_outputs['outputs_upper'],'outputs_lower': inference_outputs['outputs_lower']}
            self.per_sample_operations(output, labels, time, event_names, masks, uncertainties)
        else:
            self.per_sample_operations(output, labels, time, event_names, masks)
        self.global_operations()
    
    @abstractmethod
    def per_sample_operations(self):
        """
        Performs per sample plot of the extreme event detection maps and the variables' saving for the global operations
        """
        pass
    
    @abstractmethod
    def global_operations(self):
        """
        Performs global plots over all the samples in the test set
        """
        pass