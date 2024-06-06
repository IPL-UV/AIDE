#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class CustomEvaluator():
    """
    Custom evaluator class
    """
    def __init__(self, config, model, dataloader):
        """Initialization of the CustomEvaluator's parameters

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
    
    def evaluate(self, inference_outputs):
        """Custom evaluation of results

        :param inference_outputs: Dictionary containing the variables to perform the visualization over 
        :type inference_outputs: dict
        """
        pass