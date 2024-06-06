#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod

class GenericBackbone(ABC):
    """Abstract base class for the generic backbone. 
    Stablishes the building functions of the specific backbones
    as abstract methods. 
    """
    @abstractmethod
    def __init__(self, config):
        pass

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def implement_model(self):
        pass

    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def test(self, model):
        pass
    
    @abstractmethod
    def inference(self, model):
        pass