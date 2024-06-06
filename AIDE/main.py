#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from utils import setup_config, validate_config
from backbones.pyodBackbone import PyodBackbone
from backbones.pytorchBackbone import PytorchBackbone

backbone_dictionary = {'OutlierDetection': 'Pyod',
                        'Classification': 'Pytorch', 
                        'ImpactAssessment': 'Pytorch'}

def main(config):
    """
    Main constructor function to run the AIDE framework
    """

    backbone = globals()[ backbone_dictionary[config['task']]+'Backbone'](config)
    backbone.load_data()
    backbone.implement_model()
    backbone.train()
    backbone.test()
    backbone.inference()
    
if __name__ == '__main__':
    
    # Setup config
    config = setup_config() 
    
    # Validate config
    validate_config(config) 
    
    # Main 
    main(config)
