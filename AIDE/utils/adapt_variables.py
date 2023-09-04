#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch

def adapt_variables(config, x, masks, labels): 
    """
    Adapts the variables to fit the dimension of the model selected

    Parameters
    ----------
    config : configuration dictionary.
    x : input data of dims 
    masks : masks locating where we have values or not. 
    labels : ground truth data.
    
    Inputs of the form: 
        1D: (batch, variables, time) 
            
        2D: (batch, variables, time, x)
            can be converted to:
            -> 1D: (batch * x, variables, time)  
            
        3D: (batch, variables, time, lat, lon) 
            can be converted to:
            -> 1D: (batch * (lat * lon), variables, time)  
            -> 2D: (batch * time, variables, lat, lon)
            

    Returns
    -------
    x : adapted input.
    masks : adapted masks.
    labels : adapted labels.
    """
    if config['data']['data_dim'] == 1:
        
        return x, masks, labels
        
    else:
        
        if config['data']['data_dim'] == 2:
            
            # INPUTS
            if config['arch']['input_model_dim'] == 1:
                
                x = torch.permute(x, dims = (0,3,1,2))  
                x = x.reshape((np.prod(list(x.size())[:2]), x.size(2), x.size(3)))
                
                labels = torch.permute(labels, dims = (0,3,1,2))  
                labels = labels.reshape((np.prod(list(labels.size())[:2]), labels.size(2), labels.size(3)))
                labels = labels.any(dim = -1).to(labels.dtype)
                
                masks = torch.permute(masks, dims = (0,3,1,2))  
                masks = masks.reshape((np.prod(list(masks.size())[:2]), masks.size(2), masks.size(3)))
                masks = masks.any(dim = -1).to(masks.dtype)
        
        if config['data']['data_dim'] == 3:
            
            # INPUTS
            if config['arch']['input_model_dim'] == 1:
                
                x = torch.permute(x, dims = (0,3,4,1,2))  
                x = x.reshape((np.prod(list(x.size())[:3]), x.size(3), x.size(4)))
                
                labels = torch.permute(labels, dims = (0,3,4,1,2))  
                labels = labels.reshape((np.prod(list(labels.size())[:3]), labels.size(3), labels.size(4)))
                labels = labels.any(dim = -1).to(labels.dtype)
                
                masks = torch.permute(masks, dims = (0,3,4,1,2))  
                masks = masks.reshape((np.prod(list(masks.size())[:3]), masks.size(3), masks.size(4)))
                masks = masks.any(dim = -1).to(masks.dtype)
                
                
            elif config['arch']['input_model_dim'] == 2: 
                
                x = torch.permute(x, dims = (0,2,1,3,4))  
                x = x.reshape((np.prod(list(x.size())[:2]), x.size(2), x.size(3), x.size(4)))
                
                labels = torch.permute(labels, dims = (0,2,1,3,4))  
                labels = labels.reshape((np.prod(list(labels.size())[:2]), labels.size(2), labels.size(3), labels.size(4)))
                
                masks = torch.permute(masks, dims = (0,2,1,3,4))  
                masks = masks.reshape((np.prod(list(masks.size())[:2]), masks.size(2), masks.size(3), masks.size(4)))
                
                if config['arch']['output_model_dim'] == 1:
                    
                    labels = labels.any(dim = -1).any(dim = -1).to(labels.dtype)
    
                    masks = masks.any(dim = -1).any(dim = -1).to(masks.dtype)
                
                
    return x, masks, labels
