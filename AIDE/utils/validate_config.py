#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def validate_config(config):
    """
    Checks for the consistency between the data dimension, 
    the dimension of the model requested and the output dimension
    
    Parameters
    ----------
    config : configuration dictionary. 

    Returns
    -------
    None.
    """
    
    if not config['arch']['user_defined']:

        # Assert compatibility between data and model input dims
        assert config['data']['data_dim'] >= config['arch']['input_model_dim'], \
            'input dimension not compatible with input model dimension'
        
        # Assert compatibility between model input and model output dims
        assert config['arch']['input_model_dim'] >= config['arch']['output_model_dim'], \
            'input model dimension not compatible with output model dimension'
        
