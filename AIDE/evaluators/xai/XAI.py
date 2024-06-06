#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from abc import ABC, abstractmethod
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import captum

from captum.attr import visualization as viz
from captum.attr import * # Saliency, InputXGradient, GuidedBackprop, IntegratedGradients, LayerGradCam, GuidedGradCam, LayerActivation, LayerAttribution, DeepLift, FeatureAblation

from utils import adapt_variables

import seaborn as sns
import pandas as pd
    
import warnings
from typing import List, Tuple, Optional, Union, Dict
import matplotlib as mpl
import matplotlib.pyplot as plt
from itertools import product
from pathlib import Path
import copy

from sklearn.metrics import * #precision_score, recall_score, f1_score, accuracy_score

eps = 1e-7 

class XAI():
    '''
    Class for handling eXplainable AI (XAI) within the toolbox. It explains the predictions of the deep learning
    models by using any attribution method from `captum.attr` such as `Saliency`, `InputXGradient`, or `IntegratedGradients`.
    It should work for all kinds of output data (categorical / regression) and all kinds of input / output
    dimensionality. However, only plots for some input / output dimensionality combinations are currently available.
    In any case, a general plot showing average overall attributions of input features vs output features should always be generated.
    It also supports aggregation of the outputs (e.g. by mean) over specific dimensions, and aggregation of the outputs
    based on a mask (e.g.: consider only predictions that have low error, consider only predictions given a certain ground
    truth class, or based on a custom compartmentalization).
    
    :param config: Global configuration dictionary for the case, containing all hyperparameters. Here we use:
        config['debug']: if >= 1, it runs the XAI only for config['debug'] samples and prints some extra info
        config['task']: 'Classification' or 'ImpactAssessment' (regression)
        config['data']['num_classes']: Number of output classes or features that have been predicted
        config['data']['features']: All feature names
        config['data']['features_selected']: Indices of the features that were selected from config['data']['features']
        config['save_path']: XAI plots will be saved in config['save_path'] / 'xai'
        config['evaluation']['xai']['params']['out_agg_dim']: None or tuple, if tuple, output dims wrt which we aggregate the output
        config['evaluation']['xai']['params']['mask']: Used for masked aggregation, it selects the aggregation mode. 
            It must be one of ["none", "events[-full]", "correctness[-full]", "labels[-full]", "custom[-full]"]
        config['evaluation']['xai']['params']['type']: Captum attribution method (str)
        config['data']['data_dim'] & config['arch']['input_model_dim']: Used by `adapt_variables`
    :param model: Pytorch model
    :param dataloader: The dataloader used by the Pytorch model
    '''
    def __init__(self, config:dict, model:torch.nn.Module, dataloader:torch.utils.data.DataLoader):
        self.debug= config.get('debug', False)
        self.config = config
        self.model = model
        self.loader = dataloader
        self.task = config['task']
        self.num_classes = 1 if self.task == 'ImpactAssessment' else config['data']['num_classes']
        self.features = self.loader.dataset.config['features']
        
        #Create directory to save figures for visualization
        self.save_path = Path(config['save_path']) / 'xai'
        self.save_path.mkdir(exist_ok=True)
        
        #Get class names and feature names
        self.feature_names= [self.loader.dataset.config['features'][i] for i in self.loader.dataset.config['features_selected']]
        self.class_names= self.loader.dataset.classes if hasattr(self.loader.dataset, 'classes') \
            and self.task != 'ImpactAssessment' else [f'Class {c}' for c in range(self.num_classes)]
        
        #Over which dims to aggregate?: int, tuple or None
        self.out_agg_dim= config['evaluation']['xai']['params'].get('out_agg_dim', None)
        
        #Use an aggregation mask? One of: ["none", "correctness[-full]", "labels[-full]", "custom"]
        #The class names change depending on the agg_mode
        self.agg_mode= config['evaluation']['xai']['params'].get('mask', 'none')
        self._update_class_names()
        
    def _update_class_names(self):
        '''
        Updates `self.class_names` so that they correspond to the new class names after applying masked aggregation
        '''
        if self.agg_mode == 'none': pass
        elif self.agg_mode == 'events-full':
            self.class_names= [f'{c1} (agg. by events: {c2})' for c1 in self.class_names for c2 in self.class_names]
        elif self.agg_mode == 'events':
            self.class_names= [f'{c1} (agg. by events: {c1})' for c1 in self.class_names]
        elif self.agg_mode == 'correctness-full':
            #E.g.: ['None (correct)', 'None (incorrect)', 'Convective storm (correct)', 'Convective storm (incorrect)']
            self.class_names= [f'{c1} (agg. by {c2} predictions)' for c1 in self.class_names for c2 in ['correct', 'incorrect']]
        elif self.agg_mode == 'correctness':
            #E.g.: ['None (correct)', 'Convective storm (correct)']
            self.class_names= [f'{c1} (agg. by {c2} predictions)' for c1 in self.class_names for c2 in ['correct']]
        elif self.agg_mode == 'labels':
            self.class_names= [f'{c1} (agg. by true label: {c1})' for c1 in self.class_names]
        elif self.agg_mode == 'labels-full':
            self.class_names= [f'{c1} (agg. by true label: {c2})' for c1 in self.class_names for c2 in self.class_names]
        elif self.agg_mode == 'custom' or self.agg_mode == 'custom-full':
            assert 'agg_classes' in config['evaluation']['xai']['params'].keys(),\
                f"config['evaluation']['xai']['params']['agg_classes'] must be a list with class names when "\
                f"using {self.agg_mode=} and a custom aggregation mask is provided"
            self.agg_classes= config['evaluation']['xai']['params']['agg_classes']
            if self.agg_mode == 'custom':
                self.class_names= [f'{c1} (agg. by custom label: {c2})' for c1,c2 in zip(self.class_names, self.agg_classes)]
            else:
                self.class_names= [f'{c1} (agg. by custom label: {c2})' for c1 in self.class_names for c2 in self.agg_classes]
        else:
            raise AssertionError(f'{self.agg_mode=} must be one of ["none", "events[-full]", "correctness[-full]", "labels[-full]", "custom[-full]"]')
        if self.debug: print(f' > Using {self.agg_mode=} with {self.class_names=}')
            
    def xai(self, events:Optional[torch.Tensor]=None):
        '''
        Performs XAI over a dataloader, saving the plots of the attributions independently for each sample
        
        :param events: tensor with the same shape as model outputs `y` indicating to what event that output corresponds
            (or 0 if it corresponds to no event). It cannot be None if agg_mode == "events[-full]"
        '''
        #Reset matplotlib config and change defaults
        mpl.rcParams.update(mpl.rcParamsDefault)    
        plt.style.use('ggplot')
        mpl.rcParams.update({'font.size': 11})
        
        #Create attributions 
        attributions, inputs, labels, predictions, masks, x_shape, y_shape= attribute(self.config, self.model, 
                                self.loader, self.num_classes, self.out_agg_dim, self.agg_mode, events=events, debug=self.debug)
        assert y_shape[-1] == len(self.class_names), f'Number of output classes after applying masking {y_shape[-1]=} '\
            f' != Number of {len(self.class_names)=} with {self.class_names=}. This is likely a bug'
                     
        #We apparently run out of colors, let's create some more just in case
        base_color_list= list(plt.rcParams['axes.prop_cycle'].by_key()['color']) #list(mpl.colors.TABLEAU_COLORS)
        color_list= base_color_list + [sns.desaturate(c, 0.2) for c in base_color_list] + \
                                      [sns.desaturate(c, 0.4) for c in base_color_list]
        
        #Visualize attributions: there are different possible visualization
        #depending on the dimensionality of the data
        for event in (pbar:=tqdm(attributions.keys())):
            pbar.set_description('Visualizing explanations')
            
            #Set a global descriptive name
            agg_name= 'agg-'+self.agg_mode if self.agg_mode != 'none' else \
                     ('agg-'+str(self.out_agg_dim) if self.out_agg_dim is not None else 'no-agg')
            
            #Visualize attributions for classes vs features aggregated over any amount of extra dimensions
            figsize_nd=(min(max(9, 17*len(self.feature_names)/15), 25), 9) #(17,9)
            fig, ax= plot_attributions_nd(attributions[event], x_shape, y_shape, self.feature_names, self.class_names, figsize=figsize_nd)
            fig.savefig(self.save_path / f'{event}_{agg_name}_nd.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
     
            #Visualize attribution over 1 input dim (e.g. time) + features
            if len(x_shape) == 2 and len(y_shape) < 3:
                a_plot, t_plot, l_plot, p_plot, m_plot= attributions[event], inputs[event], labels[event], predictions[event], masks[event]
                
                #If y_shape == 2, we cannot plot the attributions for all output timesteps, so we select the timestep where the first true event is happening
                if len(y_shape) == 2:
                    timestep= (l_plot!=0).argmax(axis=0).max()
                    if m_plot is None: m_plot= np.zeros_like(p_plot) #We show this selection with the mask
                    m_plot[timestep]= 1
                    a_plot= a_plot[timestep]
                    
                #If there are no timesteps in the predictions, just create fake timesteps and position the predicitons at the end
                if len(p_plot.shape) == 1:
                    t=t_plot.shape[0]
                    l_plot, p_plot= event_at_positon(l_plot, t, position='end'), event_at_positon(p_plot, t, position='end')
                    if m_plot is None:
                        m_plot= np.zeros_like(p_plot) #We show this selection with the mask
                        m_plot[-1]= 1
                    else:
                        m_plot= event_at_positon(m_plot, t, position='end')

                #Compute automatic figsize
                figsize=(min(max(8, 15*x_shape[0]/100), 30), min(max(8, 15*x_shape[-1])/10, 20)) #(15,15)
                fig, axes= plot_attributions_1d( a_plot, t_plot, l_plot, p_plot, m_plot, 
                                                 self.feature_names, self.class_names,
                                                 color_list=color_list,
                                                 figsize=figsize, dpi=200, outlier_perc=1, 
                                                 attr_factor=0.5, alpha=0.5, margin_perc=50., 
                                                 kind='stacked', attr_baseline='feature',
                                                 names_position='top',
                                                 is_classification=True if self.task == 'Classification' else False,
                                                 title=f'Attributions ({self.agg_mode=}' + ('' if len(y_shape) == 1 else f', explaining output {timestep=}') + ')')
                fig.savefig(self.save_path / f'{event}_{agg_name}_1d.png', dpi=300, bbox_inches='tight')
                plt.close(fig)
                
            #TODO: Visualize attributions over 2 input dims (e.g. x & y) + features
            if len(x_shape) == 3:
                #print(f'NotImplemented: Attribution plotting for {len(x_shape)=} is not yet implemented')
                #plot_attributions_3d()
                pass
            
            #TODO: Visualize attributions over 3 input dims (e.g. x, y & t) + features
            if len(x_shape) == 4:
                #print(f'NotImplemented: Attribution plotting for {len(x_shape)=} is not yet implemented')
                #plot_attributions_4d()
                pass
                
            if len(x_shape) > 4:
                #print(f'Attribution plotting for {len(x_shape)=} is not supported!')
                pass
            
        return attributions, inputs
    
def attribute(config:dict, model:torch.nn.Module, dataloader:torch.utils.data.DataLoader, num_classes:int, 
              out_agg_dim:Optional[tuple], agg_mode:str, events:Optional[torch.Tensor]=None, debug:bool=False
              ) -> Tuple[Dict[str,np.ndarray], Dict[str,np.ndarray], Dict[str,np.ndarray], Dict[str,np.ndarray], List[int], List[int]]:
    '''
    Attribute over an arbitrary amount of input and output dimensions. It also supports aggregation 
    of the outputs (e.g. by mean) over specific dimensions, and aggregation of the outputs based on a mask 
    (e.g.: consider only predictions that have low error, consider only predictions given a certain ground
    truth class, or based on a custom compartmentalization).
    
    :param config: Global configuration dictionary for the case, containing all hyperparameters. Here we use:
        config['evaluation']['xai']['params']['out_agg_dim']: None or tuple, if tuple, output dims wrt which we aggregate the output
        config['evaluation']['xai']['params']['type']: Captum attribution method (str)
        config['data']['data_dim'] & config['arch']['input_model_dim']: Used by `adapt_variables`
    :param model: Pytorch model
    :param dataloader: The dataloader used by the Pytorch model
    :param num_classes: number of output classes / features for the problem
    :param out_agg_dim: tuple of dimensions wrt which we aggregate the output, or None, to perform no dimension-wise aggregation
    :param agg_mode: Used for masked aggregation, it selects the aggregation mode. 
            It must be one of ["none", "events[-full]", "correctness[-full]", "labels[-full]", "custom[-full]"]
    :param events: tensor with the same shape as model outputs `y` indicating to what event that output corresponds
        (or 0 if it corresponds to no event). It cannot be None if agg_mode == "events[-full]"
    :param debug: if >= 1, it runs the XAI only for `debug` samples and prints some extra info
    
    :return: tuple containing:
        Dictionaries with a key for every attributed event with numpy arrays:
            attributions: array with shape ([out x, out y, out t], out classes, [in x, in y, in t], in features)
            inputs: array with shape ([in x, in y, in t], in features)
            labels: array with shape ([out x, out y, out t], [out classes])
            predictions: array with shape ([out x, out y, out t], out classes)
        List with the shape of x and y after processing:
            x_shape: list containing [[in x, in y, in t], in features]
            y_shape: list containing [[out x, out y, out t], out classes]
    '''
    # GPs
    final_activation = config['implementation']['loss']['activation']['type']
    
    #If mask is provided, out_agg_dim is ignored, and the mask is used for aggregation instead
    if agg_mode != 'none' and out_agg_dim is not None:
        print(f'Warning: if {agg_mode=} != "none", then {out_agg_dim=} will be ignored')
    
    #Prepate agg_y params and load XAI method
    agg_y_kwargs= dict(out_agg_dim=out_agg_dim, num_classes=num_classes, cross_agg= agg_mode.endswith('-full'))
    xai_method = globals()[config['evaluation']['xai']['params']['type']]\
                    (lambda x, *args: agg_y(model.GP.likelihood(model(x)).mean.mean(0).unsqueeze(-1) \
                     if final_activation in ['DeepGP'] else model(x), *args, **agg_y_kwargs))
    
    #Iterate over samples and output dimensions (after aggregation)
    model.eval()
    y_shape= None
    with torch.no_grad():
        inputs, attributions, predictions, labels, masks= {}, {}, {}, {}, {}
        for index, sample in enumerate((pbar := tqdm(dataloader))): #It over events
            pbar.set_description('Explaining Dataloader')

            if not 'masks' in sample.keys(): sample['masks'] = torch.ones(sample['x'].shape)
            if not 'event_name' in sample.keys(): event_name = index*sample['x'].shape[0] + np.arange(sample['x'].shape[0])
            else: event_name = sample['event_name']

            #Get batch data and prediction
            x, _, y_labels = adapt_variables(config, sample['x'], sample['masks'], sample['labels'])
            if len(y_labels.shape) == 1: y_labels = y_labels.unsqueeze(dim=1)
            y = model.GP.likelihood(model(x)).mean.mean(0).unsqueeze(-1) \
                if final_activation in ['DeepGP'] else model(x)     
            
            #If labels seem to be int-encoded, try to one-hot encode them
            #We need to move the output channel dimension at the end for this to work
            #A new dimension will be created at the end containing the one-hot encoded labels
            #Then we move it back to position 1
            if not y.shape == y_labels.shape:
                if debug and index == 0: f' > {y.shape=} != {y_labels.shape}. Attempting to one-hot encode y_labels'
                y_labels= torch.eye(y.shape[1])[y_labels.swapaxes(1,-1)].swapaxes(1,-1)
                assert y.shape == y_labels.shape, \
                    f'An attempt was made to one hot encode {y_labels.shape=} to make it like {y.shape=}, but something failed'

            #Create aggregation mask, which is just like y, but with (possibly) a different number of 
            #output channels (the agg. channels)
            mask= get_agg_mask(agg_mode, y, y_labels, sample, events=events[[index]] 
                               if agg_mode=='events' or agg_mode == 'events-full' else None)

            #y_shape after aggregation (if used)
            if y_shape is None: 
                assert y.shape[1] in [num_classes, 1], f'Output classes must be located in axis 1 of y' 
                y_shape = agg_y(y, mask=mask, **agg_y_kwargs).shape[1:][::-1]
            x_shape = x[0].T.shape

            #Create output arrays
            if debug and index == 0:
                print('XAI shapes before aggregation:')
                print(f' > {x.shape=}\n > {y.shape=}\n > {y_labels.shape=}')
                if mask is not None: print(f' > {mask.shape=}')
            
            for e in event_name:
                attributions[e] = np.zeros([*y_shape,*x_shape]) 
                inputs[e] = x[0].T.detach().cpu().numpy()
                y= torch.concatenate([1-y, y], axis=1) if y.shape[1]==1 and num_classes==2 else y
                predictions[e] = torch.squeeze(y).T.detach().cpu().numpy() \
                    if mask!=None else agg_y(y, mask, **agg_y_kwargs)[0].T.detach().cpu().numpy()
                #torch.squeeze(y*mask).T.detach().cpu().numpy() #agg_y(y, mask, **agg_y_kwargs)[0].T.detach().cpu().numpy()
                labels[e] = torch.squeeze(y_labels).T.detach().cpu().numpy() \
                    if mask!=None else agg_y(y_labels, mask, **agg_y_kwargs)[0].T.detach().cpu().numpy()
                masks[e] = torch.squeeze(mask).T.cpu().numpy().astype('int') \
                    if mask!=None else None
            
            if debug and index == 0:
                print('XAI shapes after aggregation:')
                print(f' > {x_shape=} ([in x, in y, in t], in features)')
                print(f' > {y_shape=} ([out x, out y, out t], out classes)')
                print(f' > {attributions[event_name[0]].shape=}'
                      ' ([out x, out y, out t], out classes, [in x, in y, in t], in features)')
                print(f' > {inputs[event_name[0]].shape=} ([in x, in y, in t], in features)')
                print(f' > {labels[event_name[0]].shape=} ([out x, out y, out t], [out classes])')
                print(f' > {predictions[event_name[0]].shape=} ([out x, out y, out t], out classes)')
                if mask is not None: print(f' > {masks[event_name[0]].shape=} ([out x, out y, out t], out classes)')

            #We atribute over all output dimensions after aggregation (if needed)
            for attr_target in product(*[range(y_dim) for y_dim in y_shape]):
                attr = xai_method.attribute(x, target=attr_target[::-1], additional_forward_args=mask)
                #Cannot directly index the array because that does not support the * operator
                for e_idx, e in enumerate(event_name):
                    attributions[e].__setitem__(attr_target, attr.detach().cpu().numpy()[e_idx].T)
            if debug and index == int(debug-1): break
                
    return attributions, inputs, labels, predictions, masks, x_shape, y_shape

def get_agg_mask(agg_mode:str, y:torch.Tensor, y_labels:torch.Tensor, sample:Dict[str, torch.Tensor], 
                 events:Optional[torch.Tensor]=None, error_threshold:float=0.5) -> torch.Tensor:
    '''
    Compute the aggregation mask according to the `agg_mode`
    
    :param agg_mode: It selects the aggregation mode. 
        It must be one of ["none", "events[-full]", "correctness[-full]", "labels[-full]", "custom[-full]"]
    :param y: Model predictions
    :param y_labels: Labels
    :param sample: The dictionary returned by the dataloader containing the batch
    :param events: tensor with the same shape as model outputs `y` indicating to what event that output corresponds
        (or 0 if it corresponds to no event). It cannot be None if agg_mode == "events[-full]"
    :param error_threshold: Threshold to be used for agg_mode == 'correctness[-full]'
    
    :return: tensor of the final mask to be used for masked aggregation
    '''
    if agg_mode == 'none':
        mask=None
    elif agg_mode == 'events' or agg_mode == 'events-full':
        #The mask selects the output predictions given the threshold for each class
        mask= torch.Tensor(events > 0).to(y.device)
        mask= torch.concatenate([torch.logical_not(torch.sum(mask,dim=1,keepdim=True)>0), mask], axis=1).bool()
        if agg_mode == 'events-full': 
            mask= torch.any(mask, dim=1, keepdim=True)
            mask= torch.concatenate([mask, ~mask], axis=1) 
    elif agg_mode == 'correctness' or agg_mode == 'correctness-full':
        #The mask selects the output predictions whose abosulute error is below a threshold
        mask= torch.abs(y-y_labels) <= error_threshold
        if agg_mode == 'correctness-full': 
            mask= torch.any(mask, dim=1, keepdim=True)
            mask= torch.concatenate([mask, ~mask], axis=1) 
    elif agg_mode == 'labels' or agg_mode == 'labels-full':
        mask= y_labels.clone().type(torch.bool)
    elif agg_mode == 'custom' or agg_mode == 'custom-full':
        #Checks
        assert 'agg_mask' in sample.keys(), \
            f'If using {agg_mode=}, the aggregation mask must be accesible as sample["agg_mask"] for '\
            f'each sample produced by the dataloader'
        if agg_mode == 'custom-full':
            assert sample['agg_mask'].shape == y.shape, f'{sample["agg_mask"].shape} must be equal to {y.shape=}'
        else:
            assert all([si == sj for i, (si, sj) in enumerate(zip(sample['agg_mask'].shape, y.shape)) if i!=1]),\
                f'{sample["agg_mask"].shape} must be equal to {y.shape=} for all axis but axis {1=}'
        #Get the mask
        mask= sample["agg_mask"]
    else:
        raise AssertionError(f'{agg_mode=} must be one of ["none", "events[-full]", "correctness[-full]", "labels[-full]", "custom[-full]"]')
        
    return mask

def agg_y(y:torch.Tensor, mask:Optional[torch.Tensor]=None, out_agg_dim:Optional[tuple]=(2, 3), 
          num_classes:int=-1, cross_agg:bool=False, agg_dim_fn=torch.mean, agg_mask_fn=torch.mean) -> torch.Tensor:
    '''
        This wrapper aggregates an output tensor over dimensions `out_agg_dim` using agg. function `agg_dim_fn`. 
        For output maps / images this makes it easier to attribute with respect to a single output scalar, as opposed 
        to individual pixel output attribution.
        
        Instead, if a `mask` is provided (not None), it uses this mask to aggregate over the classes,
        either by simple direct masking if `cross_agg=False` where the mask just selects some
        wanted pixels from each output class, or by generating all possible combinations of the output
        classes with the masking classes if `cross_agg=True`. In this last case, the new number 
        of classes is the product of original_classes x aggregation_classes. 
        Samples selected by the mask are then aggregated using `agg_mask_fn`
        
        It also expands the output channel dimension if it has only a size of 1 and `num_classes`>1.
        E.g.: it expands from binary classification to 2-class multiclass output
        
        Note: Output classes must be located in dimension 1 of y, and for masked aggregation,
        dimension 1 must have a size of 1
        
        :param y: Model predictions or labels to be aggregated
        :param mask: If provided (not None), mask to aggregate over the classes
        :param out_agg_dim: None or tuple, if tuple, output dims wrt which we aggregate the output
        :param num_classes: Number of output classes or features that have been predicted
        :param cross_agg: If True, generate all possible combinations of the output classes with the masking classes
        :param agg_dim_fn: pytorch function used to aggregate `y` if not using masked aggregation. It must accept dim param
        :param agg_mask_fn: pytorch function used to aggregate `y` if using masked aggregation. It must accept axis param
        
        :return: aggregated y
    '''     
    #Expand y and mask if there is a single output class, but there should be two
    y= torch.concatenate([1-y, y], axis=1) if y.shape[1]==1 and num_classes==2 else y
    
    #Mask or maskless aggregation
    if mask is not None: #Mask aggregation. If mask is provided, out_agg_dim is ignored
        mask= torch.concatenate([~mask, mask], axis=1) if mask.shape[1]==1 and num_classes==2 else mask
        # assert y.shape[0] == 1, f'For now, first dimension of y (batch dim) must be 1. Found: {y.shape[0]}'
        if cross_agg:
            y= torch.concatenate([agg_mask_fn(y[:,[c1]][mask[:,[c2]]].reshape(y.shape[0],-1), axis=-1, keepdims=True)
                                  for c1 in range(y.shape[1]) for c2 in range(mask.shape[1])], axis=1)
        else:
            y= torch.concatenate([agg_mask_fn(y[:,[c1]][mask[:,[c1]]].reshape(y.shape[0],-1), axis=-1, keepdims=True) 
                                  for c1 in range(y.shape[1])], axis=1)
    elif out_agg_dim is not None: #Maskless aggregation. 
        y= agg_dim_fn(y, dim=out_agg_dim, keepdim=False)
    else: pass #No aggregation
        
    return y
    
def plot_attributions_nd(data:np.ndarray, x_shape:List[int], y_shape:List[int], 
                         feature_names:List[str], class_names:List[str], figsize:Tuple[float]=(17,9)
                         ) -> Tuple[mpl.figure.Figure, plt.axis]:
    '''
        Plots average attributions over all dimensions except for output classes and input features
        It should work for any kind of model and input / output dimensionality
        
        :param data: attributions array with shape ([out x, out y, out t], out classes, [in x, in y, in t], in features)
        :param x_shape: List [[in x, in y, in t], in features]
        :param y_shape: List [[out x, out y, out t], out classes]
        :param feature_names: List with the names of the input features
        :param class_names: List with the names of the output classes / features
        :param figsize: figsize to pass to plt.subplots
        
        :return: Matplotlib figure and ax
    '''
    #Build multiindex df
    row_names= [f'Output dim {i}' for i in range(len(y_shape)-1)] + ['Output class']
    row_names+= [f'Input dim {i}' for i in range(len(x_shape)-1)] + ['Input feature']
    rows= [list(range(y_i)) for y_i in y_shape[:-1]] + [class_names]
    rows+= [list(range(x_i)) for x_i in x_shape[:-1]] + [feature_names]
    attr_df= get_multiindex_df(data=data, rows=rows, columns= [], 
                               row_names=row_names, column_names= [], default_name='Attributions')

    #These transformations are needed for plotting
    attr_df_ri= attr_df.reset_index()
    attr_df_ri.columns= attr_df_ri.columns.to_flat_index()
    attr_df_ri= attr_df_ri.rename({c:c[0] for c in attr_df_ri.columns}, axis=1)

    #Plot and save
    fig, ax= plt.subplots(figsize=figsize)
    sorted_input_features= copy.copy(feature_names)
    sorted_input_features.sort(key=lambda f: -np.abs(attr_df_ri.loc[attr_df_ri['Input feature']==f, 'Attributions']).mean())
    sns.barplot(attr_df_ri, ax=ax, x='Input feature', y='Attributions', hue='Output class', order=sorted_input_features)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=12, ha='center')
    ax.grid(True)
    ax.set_title('Attribution averaged (with CI) over all other dimensions')
    return fig, ax
    
def plot_attributions_1d(attributions:np.ndarray, #(out classes, in t, in features)
                         inputs:np.ndarray, #(in t, in features)
                         labels:np.ndarray, #(out t, [out classes]) out classes is optional
                         predictions:np.ndarray, #(out t, out classes)
                         masks:np.ndarray, #(out_t, out_classes)
                         feature_names:List[str], #List of in features names
                         class_names:List[str], #List of out classes names
                         plot_classes_predictions:Optional[List[str]]=None, 
                         #Classes to plot for predictions. If None, first class is ignored
                         plot_classes_attributions:Optional[List[str]]=None, 
                         #Classes to plot for attributions. If None, first class is ignored
                         figsize:Tuple[float]=(10,10), #Matplotlib figsize
                         #out t wrt which attributions are plotted. If None, use out t where first event occurs
                         color_list=list(mpl.colors.TABLEAU_COLORS),
                         title:Optional[str]=None,
                         outlier_perc:float=2., #Get rid of N% of outliers
                         margin_perc:float=25., #Add some % of margin to feature plots
                         alpha:float=0.8, 
                         attr_factor:float=0.5,
                         attr_baseline:str='middle', #One of {'middle', 'feature'}
                         kind:str='stacked', #One of {'stacked', 'sidebyside'}
                         names_position:str='left', #One of {'left', 'top'}
                         label_fontsize:float=11,
                         is_classification:bool=True,
                         **kwargs) -> Tuple[mpl.figure.Figure, plt.axis]:
    '''
    Plot 1D attributions (e.g. of a inputs) given an output timestep.
    If you want to see absolute attributions, pass np.abs(attributions) instead.
    
    :param attributions: array with shape (out classes, in t, in features)
    :param inputs: array with shape (in t, in features)
    :param labels: array with shape (out t, [out classes]) out classes is optional
    :param predictions: array with shape (out t, out classes)
    :param masks: array with shape (out t, out classes)
    :param feature_names: List of in features names
    :param class_names: List of out classes names
    :param plot_classes_predictions: Classes to plot for predictions. If None, first class is ignored 
        if there are more than 2 classes
    :param plot_classes_attributions: Classes to plot for attributions. If None, first class is ignored 
        if there are more than 2 classes
    :param figsize: Matplotlib figsize
    :param timesteps: Global x values
    :param timestep: output timestep with respect to which which attributions are plotted. 
        If None, use the first timestep where the ground truth output class != 0
    :param color_list: list of Matplotlib-compatible colors
    :param title: title of the plot
    :param outlier_perc: Get rid of N% of attribution outliers
    :param margin_perc: Add some % of margin to feature plots
    :param alpha: transparency of the bars
    :param attr_factor: multiply attribution values by this number (e.g., if 0.5, the attribution can only
        take maximum 50% of the height of the plot. Recommended: 0.5 for signed attributions, and 1
        for absolute attributions (this is done by default)
    :param attr_baseline: Where to place the attribution bars with respect to the plot.
        attr_baseline='middle': start attribution bars from the middle of the plot
        attr_baseline='feature': start attribution bars from the feature value for that timestep
    :param kind: How to plot attributions: 
        kind='stacked': plot attributions for each class stacked on top of each other
        kind='sidebyside': plot attributions for each class side by side
    :param names_position: Where to position the names of the features
        names_position='left': position them to the left of the plot
        names_position='top': position them to the top of the plot
    :param label_fontsize: font size of the labels
    :param is_classification: whether it is a classification or a regression problem
    
    :param *kwargs: kwargs to be passed to plt.subplots
    
    :return: Matplotlib figure and ax
    '''   
    #General checks
    #assert len(class_names) > 1, f'There must be at least 2 classes, found {len(class_names)=}'
    if is_classification and not np.all(labels.astype(int) == labels): 
        warnings.warn(f'The problem has been defined as {is_classification=}. Are you sure it is?')
    to= labels.shape[0] if len(labels.shape) > 1 else 1
    assert len(attributions.shape) == 3, f'{attributions.shape=} != 3 (out classes, in t, in features)'
    co, ti, fi= attributions.shape
    assert inputs.shape == (ti, fi), f'{inputs.shape=} != {(ti, fi)=} (in t, in features)'
    assert len(feature_names) == fi, f'{len(feature_names)=} != {fi=} (in features)'
    assert len(class_names) == co, f'{len(class_names)=} != {co=} (out classes)'
    assert 0 <= outlier_perc <= 100, f'{outlier_perc=} not in range [0, 100]'
    if plot_classes_predictions is None: 
        if is_classification and co > 2: plot_classes_predictions= class_names[1:] #Ignore class 0 by default
        else: plot_classes_predictions= class_names
    assert set(plot_classes_predictions).issubset(set(class_names)), f'{plot_classes_predictions=} must be in {class_names=}'
    if plot_classes_attributions is None: 
        if is_classification and co > 2: plot_classes_attributions= class_names[1:]
        else: plot_classes_attributions= class_names
    assert set(plot_classes_attributions).issubset(set(class_names)), f'{plot_classes_attributions=} must be in {class_names=}'
    assert kind in ['stacked', 'sidebyside'], f'{kind=} must be in {["stacked", "sidebyside"]}'
    assert attr_baseline in ['middle', 'feature'], f'{attr_baseline=} must be in {["middle", "feature"]}'
    assert names_position in ['left', 'top'], f'{attr_baseline=} must be in {["left", "top"]}'
    assert len(color_list) >= len(plot_classes_predictions) and len(color_list) >= len(plot_classes_attributions),\
        f'Not enough colors in {len(color_list)=} for {len(plot_classes_predictions)=} or {len(plot_classes_attributions)=}'
    
    #Data processing
    timesteps= np.arange(0, ti)
    #timesteps_out = np.arange(0, to)
    cl_idx_predictions= np.array([class_names.index(c) for c in plot_classes_predictions])
    cl_idx_attributions= np.array([class_names.index(c) for c in plot_classes_attributions])
    
    #Set attributions to somewhere between 0 & 1
    is_attr_abs= not np.any(attributions < 0) #Attr are absolute if there is none below zero
    if is_attr_abs and attr_factor != 1.: 
        attr_factor= 1.
        warnings.warn('If attributions are absolute, attr_factor is automatically set to 1')
    attr_min, attr_max= np.percentile(attributions, [outlier_perc/2, 100-outlier_perc/2])
    attr= np.copy(attributions)
    attr[attr < attr_min]= attr_min
    attr[attr > attr_max]= attr_max
    attr/= (attr_max - attr_min + eps)
        
    #Build figure
    fig, axes = plt.subplots(figsize=figsize, nrows=fi + 3 if (np.array(masks)!=None).all() else fi + 2, sharex=True, **kwargs)
    axes= np.array([axes]) if not isinstance(axes, np.ndarray) else axes
    x_unit= timesteps[1]-timesteps[0] #Assume timesteps are equally spaced
    bar_width= 0.9 * x_unit if len(timesteps) > 1 else 0.9
    line_kwargs= dict(colors='gray', label='', linewidth=0.5, zorder=-1)
    
    #We define a function to plot a single feature + class on an mpl axis
    def plot_feature(ax:plt.axis, cl_idx:List[float], cl_names:List[str], absolute:bool=False, show_legend:bool=False,
                     ylabel:str='', x:Optional[np.ndarray]=None, y:Optional[np.ndarray]=None, y_is_attr:bool=False, ylim:Tuple[float]=None):
        '''
        Plot on `ax` a scalar feature x and/or a category-like/attribution feature y
        
        :param ax: axis on which to plot
        :param cl_idx: array or list of class indices, to be used to index over y[:,i]
        :param cl_names: class names associated with `cl_idx`, referring to y
        :param absolute: whether the attribution data is absolute (>=0) or not
        :param show_legend: whether to plot the legend
        :param ylabel: label of the axis, i.e. name of the feature
        :param x: array of scalar features, with feature name associated `label`
        :param y: array of categorical features or attribution features, with classes / features
            associated to them in `cl_idx` and `cl_names`
        :param y_is_attr: wether y is an attribution or a categorical feature (possibly with probabilities)
        :param ylim: y limits of the plot. If None, they are computed automatically from x.
            If None and x also None, they are set to (0,1)
        '''                   
        #Set limits
        if ylim is not None:
            ymin, ymax = ylim[0], ylim[1]
        elif x is not None:
            sp= margin_perc/100 * (np.max(x) - np.min(x))
            ymin, ymax= np.min(x) - sp, np.max(x) + sp
        elif not is_classification:
            sp= margin_perc/100 * (np.max(y) - np.min(y))
            ymin, ymax= np.min(y) - sp, np.max(y) + sp
        else:
            ymin, ymax= 0, 1
        if y_is_attr: y*= (ymax - ymin) * attr_factor
        
        #Plot categorical
        if y is not None: 
            #Decide where the categorical plot starts: either bottom, middle, or at the feature value
            if not y_is_attr:
                baseline= np.zeros(ti)
            elif absolute: #If using absolute attributions, always start from bottom
                baseline= np.zeros(ti) + ymin
            elif attr_baseline == 'feature': #Start from features if they exist, otherwise from the bottom
                baseline= x if x is not None else np.zeros(ti) + ymin
            elif attr_baseline == 'middle': #Start from the middle
                baseline= np.zeros(ti) + ymin + (ymax - ymin)/2
                ax.hlines(ymin + (ymax - ymin)/2, timesteps[0], timesteps[-1], **line_kwargs) 
            else: 
                raise AssertionError(f'Unknown {attr_baseline=}')
                 
            if kind == 'stacked':
                #There are two bottoms: one for the data going up, and one for the data going down
                bottom_up, bottom_down= np.copy(baseline), np.copy(baseline)
                for cl_i, cl_name in zip(cl_idx, cl_names):
                    going_up= y[:,cl_i] >= 0 #Boolean matrix idicating which data goes up now
                    bottom= np.where(going_up, bottom_up, bottom_down)
                    ax.bar(timesteps, y[:,cl_i], bar_width, label=cl_name, 
                           bottom=bottom, color=color_list[cl_i], alpha=alpha)
                    #Update either bottom_up or bottom_down accordingly
                    bottom_up[going_up]= bottom_up[going_up] + y[going_up, cl_i]
                    bottom_down[~going_up]= bottom_down[~going_up] + y[~going_up, cl_i]

            elif kind == 'sidebyside':
                class_bar_width= bar_width / len(cl_names)
                for i, (cl_i, cl_name) in enumerate(zip(cl_idx, cl_names)):
                    ax.bar(timesteps + i * class_bar_width, 
                           y[:,cl_i], class_bar_width, label=cl_name, 
                           bottom=baseline, color=color_list[cl_i], alpha=alpha)
            else:
                raise AssertionError(f'Unknown {kind=}')
            
        #Plot feature
        if x is not None:
            if not y_is_attr: #If it is not an attribution line, plot every class
                for cl_i, cl_name in zip(cl_idx, cl_names):
                    ax.plot(timesteps, x[:,cl_i], label=cl_name if y is None else None, 
                            color=ax.set_prop_cycle(color=color_list[cl_i]))
            else: #If it is an attribution line, it is just one feature, plot it
                ax.plot(timesteps, x, color=ax.set_prop_cycle(color=color_list[cl_idx[0]]))                
        
        #Set ax properties
        if show_legend: ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
        ax.set_ylim(ymin=ymin, ymax=ymax)
        ax.set_xlim([timesteps[0] - x_unit/2, timesteps[-1] + x_unit/2])
        if names_position == 'left': 
            ax.set_ylabel(ylabel, fontsize=label_fontsize)
            ax.yaxis.set_label_coords(-0.1,0.5)
        elif names_position == 'top':
            ax.set_title(ylabel, fontsize=label_fontsize, pad=2)
        else:
            raise AssertionError(f'Unknown {names_position=}')
        if x is not None: ax.grid(True, zorder=-10, linestyle='dashed', linewidth=0.5)
      
    #Plot actual and predicted class
    use_bar_plot= predictions[:-1].sum() == 0 #Use bar plot if we extended the pred. with zeros
    plot_feature(axes[0], cl_idx_predictions, plot_classes_predictions,
                 absolute=True, show_legend=True, ylabel='GT', 
                 x=None if use_bar_plot else labels,
                 y=labels if use_bar_plot else None,
                 ylim=(0,1) if is_classification else None)
    plot_feature(axes[1], cl_idx_predictions, plot_classes_predictions,
                 absolute=True, show_legend=False, ylabel='Predicted', 
                 x=None if use_bar_plot else predictions,
                 y=predictions if use_bar_plot else None,
                 ylim=(0,1) if is_classification else None) 
    
    # Plot attribution mask
    if (np.array(masks)!=None).all():
        plot_feature(axes[2], cl_idx_predictions, plot_classes_predictions,
                     absolute=True, show_legend=False, ylabel='Attr. mask', 
                     y=masks, ylim=(0,1))     
    
    #Plot input features
    for f, ax in enumerate(axes[3:] if (np.array(masks)!=None).all() else axes[2:]):       
        plot_feature(ax, cl_idx_attributions, plot_classes_attributions,
                     absolute=is_attr_abs, show_legend=f==0, 
                     ylabel=feature_names[f], 
                     x=inputs[:,f], y=attr[...,f].T, y_is_attr=True)
        
    #Figure-wide configuration
    axes[-1].set_xticks(timesteps[::(ti//30 if ti//30 else 1)])
    title= 'Attributions' if title is None else title
    fig.suptitle(title, y=0.91, size=13)
    fig.subplots_adjust(hspace=0.4 if names_position == 'top' else 0.15)
    
    return fig, axes

def get_multiindex_df(data:np.ndarray, rows:List[List[str]], columns:List[List[str]], 
                      row_names:Optional[List[str]]=None, column_names:Optional[List[str]]=None, 
                      default_name:str='values') -> pd.DataFrame:
    '''
    Builds a multiindex + multicolumn pandas dataframe. For instance, from a `data` matrix of
    shape (a, b, c, d), if len(rows) = 2, and len(columns) = 2, the output dataframe will have
    two column levels, two index levels, and a x b x c x d total rows. It is assumed that
    rows are taken from the first dimensions of `data`, and then columns are taken from the remining 
    dimensions
    
    :param data: array with a shape that is consistent with the rows + columns provided
    :param rows: a list of lists that will be used to build a MultiIndex, optionally empty list.
        For instance, the list at position 0 will contain a label for each of the features of data
        in the 0th dimension.
    :param columns: a list of lists that will be used to build a MultiIndex, optionally empty list
        For instance, the list at position 0 will contain a label for each of the features of data
        in the 0th dimension.
    :param row_names: a list of row names for the final DataFrame, len(row_names) = len(rows)
    :param column_names: a list of column names for the final DataFrame, len(column_names) = len(column_names)
    :param default_name: the default name to use for row and columns if they are not provided
        
    :return: MultiIndex pd.DataFrame containing the data
    '''
    shape= data.shape
    if not rows:
        rows= [[default_name]]
        row_names=None
        shape= (1, *shape)
    if not columns:
        columns= [[default_name]]
        column_names=None
        shape= (*shape, 1)
    
    row_index = pd.MultiIndex.from_product(rows, names=row_names)
    col_index = pd.MultiIndex.from_product(columns, names=column_names)
    row_lens= [len(r) for r in rows]
    column_lens= [len(r) for r in columns]
    assert (*row_lens, *column_lens) == shape, f'{[*row_lens, *column_lens]=} != {shape=} '
    return pd.DataFrame(data.reshape(np.product(row_lens), np.product(column_lens)), 
                        index=row_index, columns=col_index)

def event_at_positon(arr:np.ndarray, t:int, position:str='end') -> np.ndarray: 
    '''
        Takes an `arr` of shape (*), and creates a new one of shape (t, *)
                
        This is used for processing arrays before passing them to `plot_attributions_1d`, in the 
        case where there is a sinle output (instead of one output for every time step)
        
        :param arr: array to transform, shape (*)
        :param t: dimensionality of the first dimension of arr after processing
        :param position: where to place the original array with respect to the final array
            position == 'end': the original `arr` is at the last position of the 0th dimension,
                and the rest of the elements are zero.
            position == 'beginning': the original `arr` is at the first position of the 0th dimension,
                and the rest of the elements are zero
            position == 'all': the original `arr` is repeated t times over t index 0
        
        :return: transformed array with shape (t, *)
    '''
    if position == 'end':
        return np.concatenate([np.zeros([t-1, *arr.shape]), arr[None]], axis=0)
    elif position == 'beginning':
        return np.concatenate([arr[None], np.zeros([t-1, *arr.shape])], axis=0)
    elif position == 'all':
        return np.concatenate([arr[None]]*t, axis=0)
    else:
        raise AssertionError(f'{position=} must be one of ["end", "beginning", "all"]')