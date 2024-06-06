#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import torch
import gpytorch
import copy

# ApproximateGP (mini-batch)
class GaussianProcessLayer(gpytorch.models.deep_gps.DeepGPLayer):
    """ Gaussian Process (GP) Layer build on top of a deep learning architecture. 
    Implements the GPyTorch objects of the GP strategy.  

    :param params: parameters to define the GP strategy, defined in the config file implementation > loss > activation 
    :type params: dict
    :param input_dims: input dimensions of the GP layer
    :type input_dims: int
    :param output_dims: output dimensions of the GP layer
    :type output_dims: int
    """
    def __init__(self, params, input_dims, output_dims):
        """Constructor method
        """
        self.params = copy.deepcopy(params)
        
        # Variational distribution
        strategy = list(self.params['variational_strategy'])[0]
        distribution = list(self.params['variational_strategy'][strategy]['variational_distribution'])[0]
        distribution_params = self.params['variational_strategy'][strategy]['variational_distribution'][distribution]
        distribution_params = {dp_key: eval(dp_value) for dp_key, dp_value in distribution_params.items()}
        if output_dims == 1:
            output_dims = None
        if output_dims is None:
            inducing_points = torch.randn(distribution_params['num_inducing_points'], input_dims)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(output_dims, distribution_params['num_inducing_points'], input_dims)
            batch_shape = torch.Size([output_dims])
        distribution_params['batch_shape'] = batch_shape
        variational_distribution = getattr(gpytorch.variational, distribution)(**distribution_params)

        # Variational strategy
        strategy_params = self.params['variational_strategy'][strategy]
        strategy_params = {sp_key: eval(sp_value) for sp_key, sp_value in strategy_params.items() if sp_key not in ['variational_distribution', 'base']}
        strategy_params['variational_distribution'] = variational_distribution
        strategy_params['inducing_points'] = inducing_points
        variational_strategy = getattr(gpytorch.variational, strategy)(self, **strategy_params)
        strategy_base = self.params['variational_strategy'][strategy]['base']
        if strategy_base:
            base_strategy = list(self.params['variational_strategy'][strategy]['base'])[0]
            base_strategy_params = self.params['variational_strategy'][strategy]['base'][base_strategy] 
            variational_strategy = getattr(gpytorch.variational, base_strategy)(variational_strategy, **base_strategy_params)
        super().__init__(variational_strategy, input_dims, output_dims)

        # Covariance/Kernel
        nested = True
        first = True
        prev_covariance = None
        prev_covariance_params = None
        while nested:
            covariance = list(self.params['covar'])[0] if first else list(nested)[0]
            covariance_params = self.params['covar'][covariance]['params'] if first else nested[covariance]['params']
            for pk, pv in covariance_params.items():
                if isinstance(pv, dict) and 'package' in pv:
                    paux = getattr(eval(covariance_params[pk]['package']),
                                   covariance_params[pk]['type'])(**{pkey: eval(pvalue) \
                                                                     for pkey, pvalue in covariance_params[pk]['params'].items()})
                    covariance_params[pk] = paux
            covariance_params['batch_shape'] = batch_shape
            if first:
                covariance_params['ard_num_dims'] = input_dims 
                self.covar_module = getattr(gpytorch.kernels, covariance)(**covariance_params)
                nested = self.params['covar'][covariance]['nested']
                first = False
            else:
                covariance_params['ard_num_dims'] = None
                self.covar_module = getattr(gpytorch.kernels, covariance)(self.covar_module, **covariance_params)
                nested = nested[covariance]['nested']
            prev_covariance = covariance
            prev_covariance_params = covariance_params
        
        # Mean
        mean_type = self.params['mean']
        if mean_type == 'constant':
            self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        else:
            self.mean_module = gpytorch.means.LinearMean(input_dims)
        
        # Output distribution
        output_distribution = list(self.params['output_distribution'])[0]
        self.output_distribution = getattr(gpytorch.distributions, output_distribution)
        self.output_distribution_params = list(self.params['output_distribution'].values())[0]
        
        # Scale to bounds
        self.grid_bounds = eval(self.params['scale_to_bounds']) if self.params['scale_to_bounds'] != False else False
        
    def forward(self, x):
        """Forward pass to get the model outputs

        :param x: input data to the GP layer
        :type x: torch.Tensor
        :return: output data from the GP layer
        :rtype:  gpytorch.distributions.multivariate_normal
        """
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        self.output_distribution_params['mean'] = mean
        self.output_distribution_params['covariance_matrix'] = covar
        return self.output_distribution(**self.output_distribution_params)
    
class DeepGP(gpytorch.models.deep_gps.DeepGP):
    """Template class for gaussian processes

    :param params: parameters to define the GP strategy, defined in the config file implementation > loss > activation 
    :type params: dict
    :param input_dims: input dimensions of the GP layer
    :type input_dims: int
    :param output_dims: output dimensions of the GP layer
    :type output_dims: int
    :param likelihood: likelihood for the GP layer, coincides with gpytorch.likelihoods
    :type likelihood: string
    """
    def __init__(self, params, input_dims, output_dims, likelihood):
        """Constructor method
        """
        layer = GaussianProcessLayer(params, 
                                     input_dims, 
                                     output_dims
        )

        super().__init__()
        
        self.layer = layer
        
        # Likelihood
        self.likelihood = getattr(gpytorch.likelihoods, 
                                  list(likelihood.keys())[0])(**list(likelihood.values())[0])
        
    def forward(self, inputs):
        """Forward pass to get the model outputs

        :param inputs: input data to the GP layer
        :type inputs: torch.Tensor
        :return: output data from the GP layer
        :rtype: gpytorch.distributions.multivariate_normal
        """
        output = self.layer(inputs)
        return output
    