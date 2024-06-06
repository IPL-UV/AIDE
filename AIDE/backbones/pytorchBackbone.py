#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import traceback
from torch.utils.data import DataLoader
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelSummary, ModelCheckpoint, EarlyStopping
from databases import *
from backbones.models import *
from evaluators import *
from backbones.genericBackbone import GenericBackbone

from torch.utils.data import WeightedRandomSampler

class PytorchBackbone(GenericBackbone):
    """Class that implements the family of PyTorch models
    and defines the generic backbone functions specific to them 

    :param config: configuration file
    :type config: dict
    """
    def __init__(self, config):
        """Constructor method
        """
        self.config = config
        self.debug = config.get('debug', False)
                
        # Build paths
        self.cp_path= os.path.join(config['save_path'], 'checkpoints')
        self.log_path= os.path.join(self.config['save_path'])
    
    def load_data(self):
        """Loads the PyTorch Datasets for training, validation and testing and
        creates the corresponding PyTorch's Dataloader classes
        """
        # Datasets
        data_train = eval(self.config['data']['name'])(self.config, period = 'train')
        data_val = eval(self.config['data']['name'])(self.config, period = 'val')
        data_test = eval(self.config['data']['name'])(self.config, period = 'test')

        self.__create_dataloaders(data_train, data_val, data_test)

    def implement_model(self):
        """Defines and implements the PyTorch model along with the  
        checkpoint and early stopping PyTorch Lightning callbacks and the tensorboard logger. 
        These are feed to the PyTorch Lightning class, "Trainer", which automates: 
        1) enabling/disabling gradients, 
        2) running the training, validation and test dataloaders
        3) calling the Callbacks at the appropriate times
        4) putting batches and computations on the correct devices
        """
        # Model
        self.model = PytorchModel(self.config, num_data_train=len(self.train_loader.dataset))

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
        """Calls for the training of the model from scratch or
        to load a pretrained model  

        :return: PyTorch model 
        :rtype: torch.nn.Module
        """
        if self.config['from_scratch']:
            return self.__train_from_scratch()
        else:
            return self.__load_model()
    
    def test(self):
        """Calls the testing stage for the model 
        """
        # Test
        self.trainer.test(self.model, self.test_loader)
    
    def inference(self, subset='test'):
        """Calls the inference stage for the model

        :param subset: period in which inference should be performed, defaults to 'test'
        :type subset: str, optional
        :return: inference results
        :rtype: dict
        """
        # Inference 
        assert subset in ['train', 'val', 'test']
        loader= {'train':self.train_loader, 'val':self.val_loader, 'test':self.test_loader}
        evaluator = PytorchEvaluator(self.config, self.model, loader)

        inference_outputs = evaluator.inference(subset)
        evaluator.evaluate(inference_outputs, subset)
        
        return inference_outputs
    
    def __create_dataloaders(self, data_train, data_val, data_test):
        """Creates the PyTorch Dataloader classes for training, validation and testing.
        The dataloaders iterate over the data defined in the datasets and
        fetch it to the PyTorch model in each step.  

        :param data_train: dataset for training
        :type data_train: torch.utils.data.Dataset 
        :param data_val: dataset for validation
        :type data_val: torch.utils.data.Dataset
        :param data_test: dataset for testing
        :type data_test: torch.utils.data.Dataset
        """
        # DataLoaders
        self.train_loader = DataLoader(data_train,
                                  batch_size = self.config['implementation']['trainer']['batch_size'], 
                                  num_workers = self.config['implementation']['data_loader']['num_workers'],
                                  shuffle=True)
                                  # sampler=sampler)
        
        self.val_loader = DataLoader(data_val, 
                                  batch_size = self.config['implementation']['trainer']['batch_size'],
                                  num_workers = self.config['implementation']['data_loader']['num_workers'])

        if 'test_batch_size' in self.config['evaluation'].keys():
            test_batch_size = self.config['evaluation']['test_batch_size']
        else:
            test_batch_size = 1
        self.test_loader = DataLoader(data_test, 
                                  batch_size = test_batch_size,
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
        print('Evaluating ' + self.config['best_run_path'])
        model =  PytorchModel.load_from_checkpoint(self.config['best_run_path'])
        return model

    def __load_model(self):
        """Tries to load a pretrained model, if not found, terminates the execution 
        """
        try:
            checkpoint = torch.load(self.config['best_run_path'])
            self.model.load_state_dict(checkpoint['state_dict'])
        except:
            print('[!] Specify a valid trained model in the configuration file field best_run_path')
            traceback.print_exc()
            sys.exit()
