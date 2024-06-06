#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torchmetrics
import torch.nn.functional as F
import gpytorch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import tsai.all as tsai_models

from utils.adapt_variables import adapt_variables
from utils.loss import *
from user_defined.models.user_models import *
from utils.metrics_pytorch import init_metrics
from .gpytorchModel import DeepGP

import matplotlib.pyplot as plt
    
class PytorchModel(pl.LightningModule):
    """Template class for deep learning architectures

    :param config: configuration file
    :type config: dict
    """    
    def __init__(self, config, num_data_train=None):
        """Constructor method
        """
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters()

        # Config
        self.config = config

        if self.config['task'] == 'Classification':
            self.num_classes = self.config['data']['num_classes']
            if self.num_classes == 2:
                self.num_classes = 1
        elif self.config['task'] == 'ImpactAssessment':
            self.num_targets = self.config['data']['num_targets']
        
        # Define model
        self.final_activation = self.config['implementation']['loss']['activation']
        self.define_model()
                
        # Loss
        if self.final_activation['type'] not in ['DeepGP']:
            self.loss = set_loss(self.config['implementation']['loss'])
        elif self.final_activation['type'] == 'DeepGP':
            # Import python package containig the loss
            loss_package = __import__(self.config['implementation']['loss']['package'], fromlist=[''])
            self.loss = getattr(loss_package, self.config['implementation']['loss']['type'])(self.GP.likelihood,
                                                                                             self.GP, 
                                                                                             num_data_train)
            self.loss = gpytorch.mlls.DeepApproximateMLL(self.loss)
            self.GP_num_training_samples = self.final_activation['num_training_samples']

        # Initialize Logger Variables
        self.loss_train = []
        self.loss_val = []

        # Initialize Test Metrics
        self.train_metrics = init_metrics(self.config)
        self.val_metrics = init_metrics(self.config)
        self.test_metrics = init_metrics(self.config)
    
    def define_model(self):              
        """Builds the model from an external file (user defined)
        or from the libraries of deep learning. Creates the auxiliary 
        parameters according to the user specifications and modifies the 
        architecture to fit the required task 
        """
        if self.config['arch']['user_defined']:

            self.model = globals()[self.config['arch']['type']](self.config)
     
        else:
            if self.config['arch']['input_model_dim'] == 1:
                
                module = globals()[self.config['arch']['type'].split('.')[0]]
                model_type = self.config['arch']['type'].split('.')[1]
                model_params = {param: self.config['arch']['params'][param] for param in set(list(self.config['arch']['params'].keys())) - set(['out_len'])}
                if self.config['arch']['params']['out_len'] != 1:
                    model_params['c_out'] = self.config['arch']['params']['c_out'] * self.config['arch']['params']['out_len']
                self.model = getattr(module, model_type)(**model_params)

            elif self.config['arch']['input_model_dim'] == 2:
                                
                if self.config['arch']['output_model_dim'] == 1:
                    aux_params = dict(pooling = 'avg', classes = self.num_classes) 
                    
                elif self.config['arch']['output_model_dim'] == 2:                    
                    aux_params = None

                if 'encoder_weights' not in self.config['arch']['params']:
                    self.config['arch']['params']['encoder_weights'] = (None)

                if 'in_channels' not in self.config['arch']['params']:
                    self.config['arch']['params']['in_channels'] =len(self.config['data']['features_selected'])

                if 'classes' not in self.config['arch']['params'] and self.config['task'] == 'Classification':
                    self.config['arch']['params']['classes'] = self.num_classes
                    
                self.model = smp.create_model(self.config['arch']['type'], **self.config['arch']['params'], aux_params = aux_params)  
                
        if self.final_activation['type'] == 'ExactGP':
            pass
        elif self.final_activation['type'] == 'DeepGP':
            self.GP = DeepGP(self.config['implementation']['loss']['activation']['params'],
                             self.config['implementation']['loss']['activation']['input_size'],
                             self.config['data']['num_targets'],
                             self.final_activation['likelihood'])

    def forward(self, x, train=False):
        """Forward pass to get the model outputs 

        :param x: input variables 
        :type x: torch.Tensor
        :return: predictions of the model
        :rtype: torch.Tensor
        """
        x = self.model(x)

        if self.config['arch']['type'].split('.')[0] == 'tsai_models':
            if self.config['arch']['params']['out_len'] != 1: 
                x = x.reshape((x.shape[0], self.config['arch']['params']['c_out'], self.config['arch']['params']['out_len']))
        
        if self.final_activation['type'] not in ['DeepGP']:
            if self.final_activation['type'] != 'linear':
                if self.final_activation['type'] == 'sigmoid':
                    x = getattr(torch.nn.functional, self.final_activation['type'])(x)
                elif self.final_activation['type'] == 'softmax':
                    x = getattr(torch.nn.functional, self.final_activation['type'])(x, dim=1)
        else:
            x_shape = x.shape
            x = torch.moveaxis(torch.moveaxis(x, 1, 0).reshape(x_shape[1], -1), 1, 0)
            
            if train and self.config['implementation']['loss']['masked']: 
                mask = self.masks.flatten()
                x = x[mask != 0]
            
            if train and self.GP_num_training_samples != -1:
                step = max(int(x.shape[0] / self.GP_num_training_samples), 1)
                x = x[::step]
            
            if self.GP.layer.grid_bounds: 
                x = gpytorch.utils.grid.ScaleToBounds(self.GP.layer.grid_bounds[0], self.GP.layer.grid_bounds[1])(x)

            x = self.GP(x)
            
        return x
    
    def shared_step(self, batch, mode):
        """Adapts the variables for the iteration, calls the forward step, 
        computes the loss and metrics and logs their value to the logger. 

        :param batch: batch of variables given by the Dataloader in the current iteration
        :type batch: dict
        :param mode: type of step: train, val or test
        :type mode: str
        :return: results of the iteration and reference variables.
            This includes the loss and the ouput and labels for the iteration. 
        :rtype: dict
        """
        x = batch['x']
        labels = batch['labels']

        if 'masks' in batch.keys():
            self.masks = batch['masks']
        else:
            self.masks = torch.ones(labels.shape)
            
        # Adapt_variables
        x, self.masks, labels = adapt_variables(self.config, x, self.masks, labels)
        masked = self.config['implementation']['loss']['masked'] 
        
        # Forward
        if self.final_activation['type'] not in ['DeepGP']:
            output = self(x)
            
            if isinstance(output, tuple):
                output = output[-1]
                    
            # Compute loss
            if 'weight' in self.config['implementation']['loss']['params'].keys():
                self.config['implementation']['loss']['params']['weight'] = torch.Tensor(self.config['implementation']['loss']['params']['weight'])
    
            if len(labels.shape) == 1: labels = labels.unsqueeze(dim=1)
            loss= self.loss(output, labels) 
    
            if masked:
                # Correct for locations where we don't have values and take the mean
                loss = loss.squeeze(dim=1)
                loss[(torch.prod(self.masks, 1)==0)] = 0
                loss = loss.sum()/(loss.numel() - torch.sum(torch.prod(self.masks, 1)==0) + 1e-7)
            else: 
                loss = loss.mean() 
            
            if self.config['task'] == 'Classification' and self.final_activation['type'] == 'linear':
                if self.num_classes == 1:
                    output = getattr(torch.nn.functional, 'sigmoid')(output)
                else:
                    output = getattr(torch.nn.functional, 'softmax')(output, dim=1)
                
        else:
            with gpytorch.settings.num_likelihood_samples(self.final_activation['settings'][mode]['num_likelihood_samples']):
                gp_output = self(x, train=True)
                if len(list(labels.shape)) == 1:
                    labels = torch.unsqueeze(labels, dim=-1)
                output_shape = labels.shape
                labels_loss = torch.moveaxis(torch.moveaxis(labels, 1, 0).reshape(output_shape[1], -1), 1, 0)
    
                if self.config['implementation']['loss']['masked']: 
                    mask = self.masks.flatten()
                    labels_loss[mask==0] = torch.nan
                    labels_loss = labels_loss[~torch.any(labels_loss.isnan(),dim=1)]
    
                if self.GP_num_training_samples != -1:
                    step = max(int(labels_loss.shape[0] / self.GP_num_training_samples), 1)
                    labels_loss = labels_loss[::step]
    
                loss = -self.loss(gp_output, labels_loss.squeeze()).mean() # -mll
                      
                output_samples = self.GP.likelihood(self(x))
                output = output_samples.mean.mean(0).squeeze()
                output = output.reshape(tuple(np.delete(output_shape,1))).unsqueeze(dim=1)
                output_lower = output_samples.confidence_region()[0].mean(0).squeeze()
                output_lower = output_lower.reshape(tuple(np.delete(output_shape,1))).unsqueeze(dim=1)
                output_upper = output_samples.confidence_region()[1].mean(0).squeeze()
                output_upper = output_upper.reshape(tuple(np.delete(output_shape,1))).unsqueeze(dim=1)
                       
        # Log loss
        self.log(f'{mode}_loss', loss, on_step = True, on_epoch = True, prog_bar = True, logger = True, 
                 batch_size=self.config['implementation']['trainer']['batch_size'])
        # Log metrics
        self.step_metrics(output.detach(), labels, mode = mode, masks=torch.prod(self.masks, 1) if masked else None)
        
        out = {'loss': loss, 'output': output.detach(), 'labels': labels}
        if self.final_activation['type'] in ['DeepGP']:
            out['output_lower'] = output_lower.detach()
            out['output_upper'] = output_upper.detach()
        return out
    
    def step_metrics(self, outputs, labels, mode, masks=None):
        """Adapts, computes and logs evaluation metrics

        :param outputs: outputs of the model
        :type outputs: torch.Tensor
        :param labels: target variables
        :type labels: torch.Tensor
        :param mode: type of step: train, val or test
        :type mode: str
        :param masks: mask indicating where we have valid
            values in the input variables and where 
            to compute the metrics, defaults to None
        :type masks: torch.Tensor, optional
        """
        for metric_name, metric in getattr(self, mode+'_metrics').items():
            adapted_outputs, adapted_labels, adapted_masks = self.adapt_variables_for_metric(metric_name, outputs, labels, masks)
            if adapted_masks !=None:
                adapted_outputs = adapted_outputs[adapted_masks==1]
                adapted_labels = adapted_labels[adapted_masks==1]
            if len(adapted_outputs) > 0:
                metric.to(self.device).update(adapted_outputs, adapted_labels)
            
    def adapt_variables_for_metric(self, metric_name, outputs, labels, masks=None):
        """Adapts the sizes of the outputs and masks to the task

        :param metric_name: name of the metric to fetch for
        :type metric_name: str
        :param outputs: outputs of the model
        :type outputs: torch.Tensor
        :param labels: target variables
        :type labels: torch.Tensor
        :param masks: mask indicating where we have valid
            values in the input variables and where 
            to compute the metrics, defaults to None
        :type masks: torch.Tensor, optional
        :return: computed metric
        :rtype: torch.Tensor
        """
        if self.config['task'] == 'Classification' and self.num_classes > 2:
            outputs = torch.movedim(outputs,1,-1).reshape(-1, self.num_classes)
            if masks !=None:
                masks = torch.movedim(masks,1,-1).reshape(-1, self.num_classes)
        else:
            outputs = outputs.reshape(-1)
            if masks !=None:
                masks = masks.reshape(-1)

        if self.config['task'] == 'Classification': labels = labels.long()

        return outputs, labels.reshape(-1), masks

    def epoch_metrics(self, mode):
        """Computes and logs average metrics. Resets each epoch.

        :param mode: type of step: train, val or test
        :type mode: str
        """
        # Compute and log average metrics
        visualize_in_prog_bar = [self.config['implementation']['trainer']['monitor']['metric']]
        for metric_name, metric in getattr(self, mode+'_metrics').items():
            if metric_name in visualize_in_prog_bar:
                prog_bar = True
            else:
                prog_bar = False
            self.log_metric(metric.compute(), mode, metric_name, prog_bar=prog_bar)

        # Reset metrics
        for metric_name, metric in getattr(self, mode+'_metrics').items():
            metric.reset()
                
    def log_metric(self, metric, mode, metric_name, prog_bar=True):
        """Logs the metric 

        :param metric: computed metric
        :type metric: torch.Tensor
        :param mode: type of step: train, val or test
        :type mode: str
        :param metric_name: name of the metric
        :type metric_name: str
        :param prog_bar: bolean to activate the progress bar, defaults to True
        :type prog_bar: bool, optional
        """
        self.log(f'{mode}_{metric_name}', metric, on_step = False, on_epoch = True, prog_bar = prog_bar, logger = True)

    def training_step(self, batch, batch_idx):
        """Training step 

        :param batch: batch of variables given by the Pytorch Dataloader
        :type batch: dict
        :param batch_idx: index of the batch 
        :type batch_idx: int
        :return: results of the step
        :rtype: dict
        """
        res = self.shared_step(batch, mode = 'train')
        self.loss_train.append(res['loss'])
        out = {'loss': res['loss'], 'output': res['output'], 'labels': res['labels']}
        if self.final_activation['type'] in ['DeepGP']:
            out['output_lower'] = res['output_lower']
            out['output_upper'] = res['output_upper']
        return out

    def on_train_epoch_end(self):
        """Calls the computation of the metrics for each epoch
        """
        self.epoch_metrics(mode = 'train')
    
    def validation_step(self, batch, batch_idx): 
        """Validation step 

        :param batch: batch of variables given by the Pytorch Dataloader
        :type batch: dict
        :param batch_idx: index of the batch 
        :type batch_idx: int
        :return: results of the step
        :rtype: dict
        """
        res = self.shared_step(batch, mode = 'val')
        self.loss_val.append(res['loss'])
        out = {'val_loss': res['loss'], 'output': res['output'], 'labels': res['labels']}
        if self.final_activation['type'] in ['DeepGP']:
            out['output_lower'] = res['output_lower']
            out['output_upper'] = res['output_upper']
        return out

    def on_validation_epoch_end(self):
        """Calls the computation of the epoch metrics and its logging
        """
        self.epoch_metrics(mode = 'val')
        if len(self.loss_train):
            self.logger.experiment.add_scalars('train_val_losses', {'train_loss': sum(self.loss_train)/len(self.loss_train), 'val_loss': sum(self.loss_val)/len(self.loss_val)}, self.global_step)
            self.loss_train, self.loss_val = ([],[])
    
    def test_step(self, batch, batch_idx):
        """Testing step 

        :param batch: batch of variables given by the Pytorch Dataloader
        :type batch: dict
        :param batch_idx: index of the batch 
        :type batch_idx: int
        :return: results of the step
        :rtype: dict
        """
        res = self.shared_step(batch, mode = 'test')        
        out = {'test_loss': res['loss'], 'output': res['output'], 'labels': res['labels']}
        if self.final_activation['type'] in ['DeepGP']:
            out['output_lower'] = res['output_lower']
            out['output_upper'] = res['output_upper']
        return out

    def on_test_epoch_end(self):
        """Calls the computation of the epoch metrics 
        """
        self.epoch_metrics(mode = 'test')
        
    def configure_optimizers(self):
        """Configures the optimizer for the training stage

        :return: optimitzer object with the definition of the training hyperparameters
        :rtype: torch.optim.Optimizer
        """
        # build optimizer
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        if self.final_activation['type'] not in ['DeepGP']:
            print(self.config['implementation']['optimizer']['params'])
            optimizer = getattr(torch.optim, self.config['implementation']['optimizer']['type'])(trainable_params, **self.config['implementation']['optimizer']['params'])
        else:
            optimizer = getattr(torch.optim, self.config['implementation']['optimizer']['type'])([{'params': self.model.parameters()},
                                          {'params': self.GP.parameters()}], **self.config['implementation']['optimizer']['params'])     

                
        return optimizer