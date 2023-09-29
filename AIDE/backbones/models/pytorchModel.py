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
from .gpytorchModel import GaussianProcessLayer
    
class PytorchModel(pl.LightningModule):
    """
    Template class for deep learning architectures
    """    
    def __init__(self, config, num_data_train=None):
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
        if self.final_activation['type'] not in ['ExactGP', 'ApproximateGP']:
            self.loss = set_loss(self.config['implementation']['loss'])
        elif self.final_activation['type'] == 'ExactGP':
            pass
        elif self.final_activation['type'] == 'ApproximateGP':
            self.likelihood = getattr(gpytorch.likelihoods, 
                                      list(self.final_activation['likelihood'].keys())[0])(**list(self.final_activation['likelihood'].values())[0])
            # Import python package containig the loss
            loss_package = __import__(self.config['implementation']['loss']['package'], fromlist=[''])
            self.loss = getattr(loss_package, self.config['implementation']['loss']['type'])(self.likelihood,
                                                                                             self.GP, 
                                                                                             num_data_train, 
                                                                                             beta=1.0, 
                                                                                             combine_terms=True)

        # Initialize Logger Variables
        self.loss_train = []
        self.loss_val = []

        # Initialize Test Metrics
        self.train_metrics = init_metrics(self.config)
        self.val_metrics = init_metrics(self.config)
        self.test_metrics = init_metrics(self.config)
    
    def define_model(self):              

        if self.config['arch']['user_defined']:

            self.model = globals()[self.config['arch']['type']](self.config)
     
        else:

            if self.config['arch']['input_model_dim'] == 1:
                
                module = globals()[self.config['arch']['type'].split('.')[0]]
                model_type = self.config['arch']['type'].split('.')[1]
                self.model = getattr(module, model_type)(**self.config['arch']['params'])

            elif self.config['arch']['input_model_dim'] == 2:
                                
                if self.config['arch']['output_model_dim'] == 1:
                    aux_params = dict(pooling = 'avg', classes = self.num_classes) 
                    
                elif self.config['arch']['output_model_dim'] == 2:                    
                    aux_params = None

                if 'encoder_weights' not in self.config['arch']['params']:
                    self.config['arch']['params']['encoder_weights'] = (None)

                if 'in_channels' not in self.config['arch']['params']:
                    self.config['arch']['params']['in_channels'] =len(self.config['data']['features_selected'])

                if 'classes' not in self.config['arch']['params']:
                    self.config['arch']['params']['classes'] = self.num_classes
                    
                self.model = smp.create_model(self.config['arch']['type'], **self.config['arch']['params'], aux_params = aux_params)  
                
        if self.final_activation['type'] == 'ExactGP':
            pass
        elif self.final_activation['type'] == 'ApproximateGP':
            self.GP = GaussianProcessLayer(self.config['implementation']['loss']['activation']['params'],
                                           self.config['implementation']['loss']['activation']['input_size'])

    def forward(self, x):
        """
        Forward pass for model prediction
        """
        
        if self.final_activation['type'] not in ['ExactGP', 'ApproximateGP']:
            x = self.model(x)
           
            if self.final_activation['type'] != 'linear':
                if self.final_activation['type'] == 'sigmoid':
                    x = getattr(torch.nn.functional, self.final_activation['type'])(x)
                elif self.final_activation['type'] == 'softmax':
                    x = getattr(torch.nn.functional, self.final_activation['type'])(x, dim=1)
        else:
            x = self.model(x)
            if self.GP.grid_bounds: 
                x = gpytorch.utils.grid.ScaleToBounds(self.GP.grid_bounds[0], self.GP.grid_bounds[1])(x)
            x = self.GP(x)
            
        return x
    
    def shared_step(self, batch, mode):
        """
        Forward pass for model training, prediction and evaluation
        """
        x = batch['x']
        labels = batch['labels']

        if 'masks' in batch.keys():
            masks = batch['masks']
        else:
            masks = torch.ones(x.shape)
            
        # Adapt_variables
        x, masks, labels = adapt_variables(self.config, x, masks, labels)
        masked = self.config['implementation']['loss']['masked'] 
        
        # Forward
        if self.final_activation['type'] not in ['ExactGP', 'ApproximateGP']:
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
                loss[(torch.prod(masks, 1)==0)] = 0
                loss = loss.sum()/(loss.numel() - torch.sum(torch.prod(masks, 1)==0) + 1e-7)
            else: 
                loss = loss.mean() 
                
            if self.config['task'] == 'Classification' and self.final_activation['type'] == 'linear':
                if self.num_classes == 1:
                    output = getattr(torch.nn.functional, 'sigmoid')(output)
                else:
                    output = getattr(torch.nn.functional, 'softmax')(output, dim=1)
        else:
            for s_key, s_val in self.final_activation['settings'][mode].items():
                getattr(gpytorch.settings, s_key)(s_val) 
            gp_output = self(x)
            loss = -self.loss(gp_output, labels).mean() # -mll
            
            with torch.no_grad():
                output_samples = self.likelihood(self(x))
                output = output_samples.mean.squeeze()
                output_lower = output_samples.confidence_region()[0].squeeze()
                output_upper = output_samples.confidence_region()[1].squeeze()
                   
        # Log loss
        self.log(f'{mode}_loss', loss, on_step = True, on_epoch = True, prog_bar = True, logger = True, 
                 batch_size=self.config['implementation']['trainer']['batch_size'])
        # Log metrics
        self.step_metrics(output.detach(), labels, mode = mode, masks=torch.prod(masks, 1) if masked else None)
        
        out = {'loss': loss, 'output': output.detach(), 'labels': labels}
        if self.final_activation['type'] in ['ExactGP', 'ApproximateGP']:
            out['output_lower'] = output_lower.detach()
            out['output_upper'] = output_upper.detach()
        return out
    
    def step_metrics(self, outputs, labels, mode, masks=None):
        """
        Compute and log evaluation metrics
        """
        for metric_name, metric in getattr(self, mode+'_metrics').items():
            adapted_outputs, adapted_labels, adapted_masks = self.adapt_variables_for_metric(metric_name, outputs, labels, masks)
            if adapted_masks !=None:
                adapted_outputs = adapted_outputs[adapted_masks==1]
                adapted_labels = adapted_labels[adapted_masks==1]
            if len(adapted_outputs) > 0:
                metric.to(self.device).update(adapted_outputs, adapted_labels)
            
    def adapt_variables_for_metric(self, metric_name, outputs, labels, masks=None):
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
        """ 
        Compute and log average metrics. Reset metrics.
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
        self.log(f'{mode}_{metric_name}', metric, on_step = False, on_epoch = True, prog_bar = prog_bar, logger = True)

    def training_step(self, batch, batch_idx):
        res = self.shared_step(batch, mode = 'train')
        self.loss_train.append(res['loss'])
        out = {'loss': res['loss'], 'output': res['output'], 'labels': res['labels']}
        if self.final_activation['type'] in ['ExactGP', 'ApproximateGP']:
            out['output_lower'] = res['output_lower']
            out['output_upper'] = res['output_upper']
        return out

    def on_train_epoch_end(self):
        self.epoch_metrics(mode = 'train')
    
    def validation_step(self, batch, batch_idx): 
        res = self.shared_step(batch, mode = 'val')
        self.loss_val.append(res['loss'])
        out = {'val_loss': res['loss'], 'output': res['output'], 'labels': res['labels']}
        if self.final_activation['type'] in ['ExactGP', 'ApproximateGP']:
            out['output_lower'] = res['output_lower']
            out['output_upper'] = res['output_upper']
        return out

    def on_validation_epoch_end(self):
        self.epoch_metrics(mode = 'val')
        if len(self.loss_train):
            self.logger.experiment.add_scalars('train_val_losses', {'train_loss': sum(self.loss_train)/len(self.loss_train), 'val_loss': sum(self.loss_val)/len(self.loss_val)}, self.global_step)
            self.loss_train, self.loss_val = ([],[])
    
    def test_step(self, batch, batch_idx):
        res = self.shared_step(batch, mode = 'test')        
        out = {'test_loss': res['loss'], 'output': res['output'], 'labels': res['labels']}
        if self.final_activation['type'] in ['ExactGP', 'ApproximateGP']:
            out['output_lower'] = res['output_lower']
            out['output_upper'] = res['output_upper']
        return out

    def on_test_epoch_end(self):
        self.epoch_metrics(mode = 'test')
        
    def configure_optimizers(self):
        """
        Configure optimizer for training stage
        """
        # build optimizer
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        if self.final_activation['type'] not in ['ExactGP', 'ApproximateGP']:
            optimizer = torch.optim.Adam(trainable_params, \
                                         lr = self.config['implementation']['optimizer']['lr'],
                                         weight_decay = self.config['implementation']['optimizer']['weight_decay'])
        else:
            optimizer = torch.optim.Adam([{'params': self.model.parameters()},
                                          {'params': self.GP.parameters()},
                                          {'params': self.likelihood.parameters()}], \
                                         lr = self.config['implementation']['optimizer']['lr'],
                                         weight_decay = self.config['implementation']['optimizer']['weight_decay'])            
        return optimizer