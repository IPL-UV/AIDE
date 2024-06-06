#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .visualizerAbstract import *


class ImpactAssessmentVisualizer1D(VisualizerAbstract):
    """
   Visualization for 1D outputs
    """

    def __init__(self, config, model, dataloader):
        """Initialization of the ClassificationVisualizer2D's parameters

        :param config: The configuration file
        :type config: dict
        :param model: PyTorch trained model for evaluation
        :type model: class: 'torch.nn.Module'
        :param dataloader: PyTorch data iterator
        :type dataloader: class: 'torch.utils.data.DataLoader'
        """
        super().__init__(config, model, dataloader)
        self.config = config
        
        print(" [!] Impact Assessment Visualizations for 1D still not implemented") 

    def per_sample_operations(self, outputs, labels, time_indexes, event_names, masks, uncertainties=None):
        pass
                
    def global_operations(self):
        pass