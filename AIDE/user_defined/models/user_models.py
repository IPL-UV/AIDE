#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class CNN2D(nn.Module): 
    """
    CNN encoder-decoder for extreme event detection
    """
    def __init__(self, config):
        super().__init__()
        
        in_channels = len(config['data']['features_selected'])
        classes = config['data']['num_classes']
        if classes == 2:
            classes = 1
        
        exp = np.floor(np.log2(in_channels)) + 1
        
        # Architecture definition
        self.encoder = nn.Sequential(nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding='same'),
                                     nn.ReLU(),
                                     nn.Conv2d(8, 16, kernel_size=3, stride=1, padding='same'),
                                     nn.ReLU())
        self.decoder = nn.Sequential(nn.Conv2d(16, 8, kernel_size=3, stride=1, padding='same'),
                                     nn.ReLU(),
                                     nn.Conv2d(8, in_channels, kernel_size=3, stride=1, padding='same'),
                                     nn.ReLU())
        self.out = nn.Conv2d(in_channels, classes, kernel_size=3, stride=1, padding='same')
    
    def forward(self, x):
        """
        Forward pass
        """
        # Pass through encoder
        x = self.encoder(x)
          
        # Pass through decoder
        x = self.decoder(x)
          
        return self.out(x)
  
    
class UD_LSTM(nn.Module):
    """
    LSTM for extreme event detection
    """
    def __init__(self, config):
        super().__init__()
        
        input_dim = np.prod(config['data']['input_size_train'])

        if config['data']['num_classes'] > 2:
            classes = config['data']['num_classes']
        else:
            classes = 1
        
        
        # Hidden dimensions
        self.hidden_dim = config['arch']['params']['hidden_dim']
        
        # Number of hidden layers
        self.n_layer = 1
        
        # LSTM definition
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, self.hidden_dim, self.n_layer, batch_first = True)
        
        # Readout layer
        self.fc = nn.Linear(self.hidden_dim, classes)
        # self.fc = nn.Linear(self.hidden_dim, int(self.hidden_dim/2))
        # self.fc2 = nn.Linear(int(self.hidden_dim/2), classes)
    
    def forward(self, x):
        """
        Forward pass
        """

        # Initialize hidden state with zeros
        #h0 = torch.zeros(self.n_layer, x.size(0), self.hidden_dim).requires_grad_()
    
        # Initialize cell state
        #c0 = torch.zeros(self.n_layer, x.size(0), self.hidden_dim).requires_grad_()
    
        # 28 time steps
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        #out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        x = x.permute(0,2,1)
        out, _ = self.lstm(x)
        
        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
        # out = self.fc2(F.relu(self.fc(out[:, -1, :])))
        out = self.fc(out[:, -1, :])
        # out.size() --> 100, 10
        return out

class UD_LSTM_IA(nn.Module):
    """
    LSTM for extreme event detection
    """
    def __init__(self, config):
        super().__init__()
        
        input_dim = np.prod(config['arch']['params']['input_size'])
        
        # Hidden dimensions
        self.hidden_dim = config['arch']['params']['hidden_dim']
        self.fc_hidden_dim = config['arch']['params']['fc_hidden_dim']
        
        # Number of hidden layers
        self.n_layer = config['arch']['params']['hidden_layers']
        
        # LSTM definition
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, self.hidden_dim, self.n_layer, batch_first = True)
        self.lstm_ln = nn.LayerNorm(self.hidden_dim)
        
        # Readout layer
        self.fc = nn.Linear(self.hidden_dim, self.fc_hidden_dim)
        self.fc_ln = nn.LayerNorm(self.fc_hidden_dim)
        self.fc_top = nn.Linear(self.fc_hidden_dim, 1)
        
        # Dropout
        self.dropout_p = config['arch']['params']['dropout_p']
        self.dropout = nn.Dropout(p=self.dropout_p)
        
        # Weight initialization
        self.apply(self._weights_init)
    
    def forward(self, x):
        """
        Forward pass
        """
        # x, mask = x
        x = x.permute(0,2,1)
        out, _ = self.lstm(x)
        
        #mask = torch.cat((mask,torch.zeros(mask.shape[0],1).to(mask.device)),axis=1)
        #tend = torch.where((mask[:,1:]-mask[:,:-1])==-1)
        
        #out = self.fc_top(self.dropout(F.relu(self.fc_ln(self.fc(self.dropout(self.lstm_ln(out[tend[0], tend[1], :])))))))
        #out = self.fc_top(self.dropout(F.relu(self.fc(self.dropout(out[tend[0], tend[1], :])))))
        out = self.fc_top(self.dropout(F.relu(self.fc(self.dropout(out[:, -1, :])))))
        return out

    def _weights_init(self, m): 
        # same initialization as keras. Adapted from the initialization developed 
        # by JUN KODA (https://www.kaggle.com/junkoda) in this notebook
        # https://www.kaggle.com/junkoda/pytorch-lstm-with-tensorflow-like-initialization
        for name, params in m.named_parameters():
            if "weight_ih" in name: 
                nn.init.xavier_normal_(params)
            elif 'weight_hh' in name: 
                nn.init.orthogonal_(params)
            elif 'bias_ih' in name:
                params.data.fill_(0)
                # Set forget-gate bias to 1
                n = params.size(0)
                params.data[(n // 4):(n // 2)].fill_(1)
            elif 'bias_hh' in name:
                params.data.fill_(0)
                                
class UD_LSTM_IA_GP(nn.Module):
    """
    LSTM for extreme event detection
    """
    def __init__(self, config):
        super().__init__()
        
        input_dim = np.prod(config['arch']['params']['input_size'])
        
        # Hidden dimensions
        self.lstm_hidden_dim = config['arch']['params']['lstm_hidden_dim']
        self.num_lstm_layers = config['arch']['params']['num_lstm_layers']
        self.fc_hidden_dim = config['arch']['params']['fc_hidden_dim']
        self.num_fc_layers = config['arch']['params']['num_fc_layers']
        self.fc_top_dim = config['arch']['params']['fc_top_dim']
        
        # LSTM definition
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, self.lstm_hidden_dim, self.num_lstm_layers, batch_first = True)
        
        # Readout layer
        self.fc = nn.ModuleList()
        for f in np.arange(self.num_fc_layers):
            self.fc.append(nn.Linear(self.lstm_hidden_dim if f==0 else self.fc_hidden_dim, self.fc_hidden_dim))
        self.fc.append(nn.Linear(self.fc_hidden_dim,self.fc_top_dim))
        
        # Dropout
        self.dropout_p = config['arch']['params']['dropout_p']
        self.dropout = nn.Dropout(p=self.dropout_p)
        
        # Weight initialization
        self.apply(self._weights_init)
    
    def forward(self, x):
        """
        Forward pass
        """
        x = x.permute(0,2,1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        for f in np.arange(self.num_fc_layers):
            out = self.fc[f](out)
            out = nn.ReLU()(out) if f==0 else self.dropout(nn.ReLU()(out))
        out = self.fc[-1](out)
        return out

    def _weights_init(self, m): 
        # same initialization as keras. Adapted from the initialization developed 
        # by JUN KODA (https://www.kaggle.com/junkoda) in this notebook
        # https://www.kaggle.com/junkoda/pytorch-lstm-with-tensorflow-like-initialization
        for name, params in m.named_parameters():
            if "weight_ih" in name: 
                nn.init.xavier_normal_(params)
            elif 'weight_hh' in name: 
                nn.init.orthogonal_(params)
            elif 'bias_ih' in name:
                params.data.fill_(0)
                # Set forget-gate bias to 1
                n = params.size(0)
                params.data[(n // 4):(n // 2)].fill_(1)
            elif 'bias_hh' in name:
                params.data.fill_(0)

class UD_LSTM_TS(nn.Module):
    """
    LSTM for extreme event detection
    """
    def __init__(self, config):
        super().__init__()
        
        input_dim = np.prod(config['data']['input_size'])

        if config['data']['num_classes'] > 2:
            classes = config['data']['num_classes']
        else:
            classes = 1
        
        # Batch size
        self.batch_size = config['implementation']['trainer']['batch_size']
        
        # Hidden dimensions
        self.hidden_dim = config['arch']['params']['hidden_dim']
        
        # Number of hidden layers
        self.n_layer = config['arch']['params']['hidden_layers']
        
        # LSTM definition
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, self.hidden_dim, self.n_layer, batch_first= True)
        # Initialize hidden state with zeros
        self.h = torch.zeros(self.n_layer, self.batch_size, self.hidden_dim).requires_grad_()
        # Initialize cell state
        self.c = torch.zeros(self.n_layer, self.batch_size, self.hidden_dim).requires_grad_()
        
        # Readout layer
        self.activation= nn.SiLU()
        self.fc = nn.Linear(self.hidden_dim, int(self.hidden_dim/2))
        self.fc2 = nn.Linear(int(self.hidden_dim/2), classes)
       
    """    
    def init_hidden(self, batch_size, nunits):
        weight = next(self.parameters())
        hidden = (weight.new_zeros(self.n_layer, batch_size, nunits),
                weight.new_zeros(self.n_layer, batch_size, nunits))
        return hidden

    def repackage_hidden(self, h):
        #Wraps hidden states in new Tensors, to detach them from their history.
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)
    """
        
    def forward(self, x):
        """
        Forward pass
        """

        # Initialize and repackage hidden states
        # self.h, self.c = self.init_hidden(self.batch_size, self.hidden_dim)
        # self.h, self.c = self.repackage_hidden((self.h, self.c))
    
        # 28 time steps
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        #out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        x = x.permute(0,2,1) #(b, c, t) -> (b, t, c)
        out, _ = self.lstm(x) 
        
        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
        out = self.fc2(self.activation(self.fc(out)))
        # out.size() --> 100, 10
        return out.permute(0,2,1) #(b, t, c) -> (b, c, t), ordering required by CE loss