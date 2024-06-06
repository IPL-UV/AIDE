#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Binary Cross Entropy
import torch
import inspect

# User defined losses
from user_defined.losses import *

def set_loss(parameters):
    """Check if loss is user defined.

    """
    if not parameters['user_defined']:
        # Import python package containig the loss
        package = __import__(parameters['package'], fromlist=[''])

        # Check if chosen loss is a class or a function module
        if inspect.isclass(getattr(package, parameters['type'])):
            loss = getattr(package, parameters['type'])(**parameters['params'])
        else:
            # If function, create class wrapper 
            loss = Loss(package, parameters)
    else:
        # Create user defined loss class
        loss = globals()[parameters['type']](**parameters['params'])

    return loss

class Loss(torch.nn.Module):
    """This class is used for wrapping for Torch loss functions.

    """
    def __init__(self, package, configuration):
        """Initialization of the loss.

        :param package: The package path of the loss function
        :type package: Str
        :param configuration: The configuration file.
        :type configuration: Dict
        """
        super().__init__()
        self.configuration = configuration
        self.loss = getattr(package, configuration['type'])
    
    def forward(self, outputs, labels):
        """ Forward function of the loss, playing the same role as __call__ does for a regular python class.

        :param outputs: Prediction/output variables by your model.
        :type outputs: Torch.Tensor
        :param labels: Target/ground-truth labels.
        :type labels: Torch.Tensor

        :return: The loss with respect to all Tensors.
        :rtype: Torch.Tensor
        """
        return self.loss(outputs, labels, **self.configuration['params'])



