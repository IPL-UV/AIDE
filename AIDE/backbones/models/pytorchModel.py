#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torchmetrics
import torch.nn.functional as F
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import tsai.all as tsai_models

from utils.adapt_variables import adapt_variables
from utils.loss import *
from user_defined.models.user_models import *
from utils.metrics_pytorch import init_metrics
    
class PytorchModel(pl.LightningModule):
    """
    Template class for deep learning architectures
    """    
    def __init__(self, config):
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
        self.define_model()
                
        # Loss
        self.loss = set_loss(self.config['implementation']['loss'])

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
                self.model = getattr(module, model_type)(**self.config['arch']['args'])

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
                

        self.final_activation = self.config['implementation']['loss']['activation']

    def forward(self, x):
        """
        Forward pass for model prediction
        """
        x = self.model(x)
       
        if self.final_activation != 'linear':
            if self.final_activation == 'sigmoid':
                x = getattr(torch.nn.functional, self.final_activation)(x)
            else:
                x = getattr(torch.nn.functional, self.final_activation)(x, dim=1)
            
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
        
        # Forward
        output = self(x)

        if isinstance(output, tuple):
            output = output[-1]
                
        # Compute loss
        if 'weight' in self.config['implementation']['loss']['params'].keys():
            self.config['implementation']['loss']['params']['weight'] = torch.Tensor(self.config['implementation']['loss']['params']['weight'])

        if len(labels.shape) == 1: labels = labels.unsqueeze(dim=1)
        loss= self.loss(output, labels) 

        masked = self.config['implementation']['loss']['masked']
        if masked:
            # Correct for locations where we don't have values and take the mean
            loss = loss.squeeze(dim=1)
            loss[(torch.prod(masks, 1)==0)] = 0
            loss = loss.sum()/(loss.numel() - torch.sum(torch.prod(masks, 1)==0) + 1e-7)
        else: 
            loss = loss.mean() 
                   
        # Log loss
        self.log(f'{mode}_loss', loss, on_step = True, on_epoch = True, prog_bar = True, logger = True, 
                 batch_size=self.config['implementation']['trainer']['batch_size'])
        # Log metrics
        self.step_metrics(output.detach(), labels, mode = mode, masks=torch.prod(masks, 1) if masked else None)
        
        return {'loss': loss, 'output': output.detach(), 'labels': labels}
    
    def step_metrics(self, outputs, labels, mode, masks=None):
        """
        Compute and log evaluation metrics
        """
        for metric_name, metric in getattr(self, mode+'_metrics').items():
            adapted_outputs, adapted_labels, adapted_masks = self.adapt_variables_for_metric(metric_name, outputs, labels, masks)
            if adapted_masks !=None:
                adapted_outputs = adapted_outputs[adapted_masks==1]
                adapted_labels = adapted_labels[adapted_masks==1]
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
        return {'loss': res['loss'], 'output': res['output'], 'labels': res['labels']}

    def on_train_epoch_end(self):
        self.epoch_metrics(mode = 'train')
    
    def validation_step(self, batch, batch_idx): 
        res = self.shared_step(batch, mode = 'val')
        self.loss_val.append(res['loss'])
        return {'val_loss': res['loss'], 'output': res['output'], 'labels': res['labels']}

    def on_validation_epoch_end(self):
        self.epoch_metrics(mode = 'val')
        if len(self.loss_train):
            self.logger.experiment.add_scalars('train_val_losses', {'train_loss': sum(self.loss_train)/len(self.loss_train), 'val_loss': sum(self.loss_val)/len(self.loss_val)}, self.global_step)
            self.loss_train, self.loss_val = ([],[])
    
    def test_step(self, batch, batch_idx):
        res = self.shared_step(batch, mode = 'test')        
        return {'test_loss': res['loss'], 'output': res['output'], 'labels': res['labels']}

    def on_test_epoch_end(self):
        self.epoch_metrics(mode = 'test')
        
    def configure_optimizers(self):
        """
        Configure optimizer for training stage
        """
        # build optimizer
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.Adam(trainable_params, lr = self.config['implementation']['optimizer']['lr'], weight_decay = self.config['implementation']['optimizer']['weight_decay'])
        return optimizer