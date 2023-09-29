#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import torch
import gpytorch
import copy

# ApproximateGP (mini-batch)
class GaussianProcessLayer(gpytorch.models.ApproximateGP):
    def __init__(self, params, input_size):
        self.params = copy.deepcopy(params)
        
        # Variational distribution
        strategy = list(self.params['variational_strategy'])[0]
        distribution = list(self.params['variational_strategy'][strategy]['variational_distribution'])[0]
        distribution_params = self.params['variational_strategy'][strategy]['variational_distribution'][distribution]
        distribution_params = {dp_key: eval(dp_value) for dp_key, dp_value in distribution_params.items()}
        if 'num_inducing_points' in list(distribution_params.keys()): distribution_params['num_inducing_points'] = distribution_params['num_inducing_points']**input_size
        variational_distribution = getattr(gpytorch.variational, distribution)(**distribution_params)

        # Variational strategy
        strategy_params = self.params['variational_strategy'][strategy]
        strategy_params = {sp_key: eval(sp_value) for sp_key, sp_value in strategy_params.items() if sp_key not in ['variational_distribution', 'base']}
        strategy_params['variational_distribution'] = variational_distribution
        if 'grid_bounds' in list(strategy_params.keys()): strategy_params['grid_bounds'] = input_size*[strategy_params['grid_bounds']]
        variational_strategy = getattr(gpytorch.variational, strategy)(self, **strategy_params)
        strategy_base = self.params['variational_strategy'][strategy]['base']
        if strategy_base:
            base_strategy = list(self.params['variational_strategy'][strategy]['base'])[0]
            base_strategy_params = self.params['variational_strategy'][strategy]['base'][base_strategy] 
            variational_strategy = getattr(gpytorch.variational, base_strategy)(variational_strategy, **base_strategy_params)
        super().__init__(variational_strategy)

        # Covariance/Kernel
        nested = True
        first = True
        prev_covariance = None
        prev_covariance_params = None
        while nested:
            covariance = list(self.params['covar'])[0] if first else list(nested)[0]
            covariance_params = self.params['covar'][covariance]['params'] if first else nested[covariance]['params']
            for pk, pv in covariance_params.items():
                if 'package' in pv:
                    paux = getattr(eval(covariance_params[pk]['package']),
                                   covariance_params[pk]['type'])(**{pkey: eval(pvalue) \
                                                                     for pkey, pvalue in covariance_params[pk]['params'].items()})
                    covariance_params[pk] = paux
            if first:
                self.covar_module = getattr(gpytorch.kernels, covariance)(**covariance_params)
                nested = self.params['covar'][covariance]['nested']
                first = False
            else:
                self.covar_module = getattr(gpytorch.kernels, covariance)(self.covar_module, **covariance_params)
                nested = nested[covariance]['nested']
            prev_covariance = covariance
            prev_covariance_params = covariance_params
        
        # Mean
        mean = list(self.params['mean'])[0]
        mean_params = list(self.params['mean'].values())[0]
        self.mean_module = getattr(gpytorch.means, mean)(**mean_params)
        
        # Output distribution
        output_distribution = list(self.params['output_distribution'])[0]
        self.output_distribution = getattr(gpytorch.distributions, output_distribution)
        self.output_distribution_params = list(self.params['output_distribution'].values())[0]
        
        # Scale to bounds
        self.grid_bounds = eval(self.params['scale_to_bounds']) if self.params['scale_to_bounds'] != False else False

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        self.output_distribution_params['mean'] = mean
        self.output_distribution_params['covariance_matrix'] = covar
        return self.output_distribution(**self.output_distribution_params)