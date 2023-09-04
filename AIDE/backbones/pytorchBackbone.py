#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from torch.utils.data import DataLoader
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelSummary, ModelCheckpoint, EarlyStopping
from databases import *
from backbones.models import *
from evaluators import *
from backbones.genericBackbone import GenericBackbone

from torch.utils.data import WeightedRandomSampler

class PytorchBackbone(GenericBackbone):
    def __init__(self, config):
        self.config = config
        self.debug = config.get('debug', False)
                
        # Build paths
        self.cp_path= os.path.join(config['save_path'], 'checkpoints')
        self.log_path= os.path.join(self.config['save_path'])
    
    def load_data(self):
        # Datasets
        data_train = eval(self.config['data']['name'])(self.config, period = 'train')
        data_val = eval(self.config['data']['name'])(self.config, period = 'val')
        data_test = eval(self.config['data']['name'])(self.config, period = 'test')

        self.__create_dataloaders(data_train, data_val, data_test)

    def implement_model(self):
        # Model
        self.model = PytorchModel(self.config)

        # Loggers
        # wandb_logger = pl_loggers.WandbLogger(save_dir = config['trainer']['save_dir'] + '/logs/', 
        #                                       project = 'template', mode = 'online')
        # Callbacks
        # Init ModelCheckpoint callback, monitoring 'val_loss'
        checkpoint_callback = ModelCheckpoint(dirpath = self.cp_path,
                                               filename = '{epoch}-{'+self.config['implementation']['trainer']['monitor']['split']+'_'+self.config['implementation']['trainer']['monitor']['metric']+':.6f}',
                                               mode = self.config['implementation']['trainer']['monitor_mode'],
                                               monitor =  self.config['implementation']['trainer']['monitor']['split']+'_'+self.config['implementation']['trainer']['monitor']['metric'],
                                               save_last = True, save_top_k = 3)
        
        early_stopping = EarlyStopping(monitor = self.config['implementation']['trainer']['monitor']['split']+'_'+self.config['implementation']['trainer']['monitor']['metric'],
                                       min_delta = 0.0, 
                                       patience = self.config['implementation']['trainer']['early_stop'], 
                                       verbose = False, mode = self.config['implementation']['trainer']['monitor_mode'], strict = True)
        
        callbacks = [checkpoint_callback, early_stopping, ModelSummary(max_depth=-1)]
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=self.log_path)

        # Trainer
        self.trainer = pl.Trainer(accumulate_grad_batches = 1, callbacks = callbacks, 
                              accelerator=self.config['implementation']['trainer']['accelerator'],
                              devices=self.config['implementation']['trainer']['devices'],
                              gradient_clip_val = self.config['implementation']['optimizer']['gclip_value'],
                              logger = [tb_logger], 
                              max_epochs = 1 if self.debug else self.config['implementation']['trainer']['epochs'],
                              reload_dataloaders_every_n_epochs = 1, 
                              val_check_interval = 1.0, 
                              enable_model_summary=True) 
    def train(self):
        if self.config['from_scratch']:
            return self.__train_from_scratch()
        else:
            return self.__load_model()
    
    def test(self):
        # Test
        self.trainer.test(self.model, self.test_loader)
    
    def inference(self, subset='test'):
        # Inference 
        assert subset in ['train', 'val', 'test']
        loader= {'train':self.train_loader, 'val':self.val_loader, 'test':self.test_loader}
        evaluator = PytorchEvaluator(self.config, self.model, loader)

        inference_outputs = evaluator.inference(subset)
        evaluator.evaluate(inference_outputs, subset)
        
        return inference_outputs
    
    def __create_dataloaders(self, data_train, data_val, data_test):
        # DataLoaders
        self.train_loader = DataLoader(data_train,
                                  batch_size = self.config['implementation']['trainer']['batch_size'], 
                                  num_workers = self.config['implementation']['data_loader']['num_workers'],
                                  shuffle=True)
                                  # sampler=sampler)
        
        self.val_loader = DataLoader(data_val, 
                                  batch_size = self.config['implementation']['trainer']['batch_size'],
                                  num_workers = self.config['implementation']['data_loader']['num_workers'])
                
        self.test_loader = DataLoader(data_test, 
                                  batch_size = 1,
                                  num_workers = self.config['implementation']['data_loader']['num_workers'])

    def __train_from_scratch(self):
        # Training
        self.trainer.fit(self.model, self.train_loader, self.val_loader)
        
        # Select best model
        list_checkpoints = [filename for filename in os.listdir(self.cp_path) if self.config['implementation']['trainer']['monitor']['split']+'_'+self.config['implementation']['trainer']['monitor']['metric'] in filename]
        assert len(list_checkpoints), f'Checkpoint list empty at {self.cp_path}'
        best_loss_value = eval(self.config['implementation']['trainer']['monitor_mode']) \
            ([float(filename.split('=')[-1][:-5]) for filename in list_checkpoints])
        best_model = [filename for filename in list_checkpoints if str(best_loss_value) in filename][0]
        self.config['best_run_path']= os.path.join(self.cp_path, best_model)
        model = self.model.load_from_checkpoint(self.config['best_run_path'])
        return model

    def __load_model(self):
        try:
            checkpoint = torch.load(self.config['best_run_path'])
            self.model.load_state_dict(checkpoint['state_dict'])
        except:
            print('[!] Specify a valid trained model in the configuration file field best_run_path')
            sys.exit()
