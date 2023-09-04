#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from databases import *
from evaluators import *
from backbones.models import *
from backbones.genericBackbone import GenericBackbone

class PyodBackbone(GenericBackbone):
    def __init__(self, config):
        self.config = config

    def load_data(self):
        self.data_train = eval(self.config['data']['name'])(self.config, period='train').__getallitems__()
        self.data_val = eval(self.config['data']['name'])(self.config, period='val').__getallitems__()
        self.data_test = eval(self.config['data']['name'])(self.config, period='test').__getallitems__()

    def implement_model(self):
        self.model = PyodModel(self.config)
    
    def train(self):
        # Training stage
        res_train = self.model.forward(self.data_train, mode='train', step_samples=self.config['arch']['step_samples_train'])
        # Validation stage
        res_val = self.model.forward(self.data_val, mode='val', step_samples=self.config['arch']['step_samples_evaluation'])
        return self.model 

    def test(self):
        # Test stage
        res = self.model.forward(self.data_test, mode='test', step_samples=self.config['arch']['step_samples_evaluation'])        

    def inference(self):
        # Inference stage
        models = {self.config['arch']['type']: self.model.get_model()}
        evaluator = PyodEvaluator(self.config, models, self.data_test)
        evaluator.evaluate()
    
    
