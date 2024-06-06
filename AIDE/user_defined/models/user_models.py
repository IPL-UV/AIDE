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
  
class CNN3D(nn.Module): 
    """
    CNN encoder-decoder for extreme event detection
    """
    def __init__(self, config):
        super().__init__()
        
        in_channels = len(config['data']['features_selected'])
        if 'num_classes' in config['data']:
            classes = config['data']['num_classes']
            if classes == 2:
                classes = 1
        else:
            classes = 1
        # Conv params
        conv_params = {'kernel_size': (3,3,3), 'stride': (1,1,1), 
                       'padding': 'same', 'padding_mode': 'reflect'}

        # Architecture definition    
        # Encoder
        self.encoder_conv_1 = nn.Conv3d(in_channels, 8, **conv_params)
        self.encoder_conv_2 = nn.Conv3d(8, 16, **conv_params)
        self.encoder_conv_3 = nn.Conv3d(16, 32, **conv_params)
        self.encoder_conv_4 = nn.Conv3d(32, 64, **conv_params)
        
        #  Normalization layers encoder
        self.encoder_bn_1 = nn.BatchNorm3d(8)
        self.encoder_bn_2 = nn.BatchNorm3d(16)
        self.encoder_bn_3 = nn.BatchNorm3d(32)
        self.encoder_bn_4 = nn.BatchNorm3d(64)
        
        # Decoder
        self.decoder_conv_4 = nn.Conv3d(64, 32, **conv_params)
        self.decoder_conv_3 = nn.Conv3d(32, 16, **conv_params)
        self.decoder_conv_2 = nn.Conv3d(16, 8, **conv_params)
        self.decoder_conv_1 = nn.Conv3d(8, classes, **conv_params)
        
        # Normalization layers decoder
        self.decoder_bn_4 = nn.BatchNorm3d(32)
        self.decoder_bn_3 = nn.BatchNorm3d(16)
        self.decoder_bn_2 = nn.BatchNorm3d(8)
        
        # Dropout
        self.dropout = nn.Dropout3d(p=0)
        
        """
        # Initializate the bias of inner layers and the output layer
        # Encoder
        torch.nn.init.constant_(self.encoder_conv_1.bias, 0)
        torch.nn.init.constant_(self.encoder_conv_2.bias, 0)
        torch.nn.init.constant_(self.encoder_conv_3.bias, 0)
        torch.nn.init.constant_(self.encoder_conv_4.bias, 0)
        # Decoder
        torch.nn.init.constant_(self.decoder_conv_4.bias, 0)
        torch.nn.init.constant_(self.decoder_conv_3.bias, 0)
        torch.nn.init.constant_(self.decoder_conv_2.bias, 0)
        torch.nn.init.constant_(self.decoder_conv_1.bias, 0)
        # Output
        torch.nn.init.constant_(self.out_conv.bias, 0)
        """
        # Activation
        self.activation  = nn.LeakyReLU()

        self._3d_pool = nn.MaxPool3d(kernel_size = (2,2,2), stride = (2,2,2))
        self._3d_upsample = nn.Upsample(scale_factor = (2,2,2), mode = 'nearest')
    
    def forward(self, x):
        """
        Forward pass
        """                                                         
        x = self.encoder_conv_1(x)                                          
        x = self.encoder_bn_1(x)                                                                     
        x = self.activation(x)                                                                                                       
        x = self._3d_pool(x)  
        x = self.dropout(x)
        skip1 = x
        
        #2
        x = self.encoder_conv_2(x)                                          
        x = self.encoder_bn_2(x)                                                                    
        x = self.activation(x)                                                 
        x = self._3d_pool(x)
        x = self.dropout(x)
        skip2 = x
        
        #3
        x = self.encoder_conv_3(x)                                          
        x = self.encoder_bn_3(x)                                                                    
        x = self.activation(x)                                                 
        x = self._3d_pool(x)
        x = self.dropout(x)

        #3
        x = self._3d_upsample(x)
        x = self.decoder_conv_3(x)                                          
        x = self.decoder_bn_3(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = x + skip2

        #2
        x = self._3d_upsample(x)
        x = self.decoder_conv_2(x)                                          
        x = self.decoder_bn_2(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = x + skip1

        #1
        x = self._3d_upsample(x)
        x = self.decoder_conv_1(x)                                          
        
        return x
    
class CNN3D_IA(CNN3D): 
    def forward(self, x):
        """
        Forward pass
        """                                                         
        x = self.encoder_conv_1(x)                                          
        x = self.encoder_bn_1(x)                                                                     
        x = self.activation(x)                                                                                                       
        x = self._3d_pool(x)  
        x = self.dropout(x)
        skip1 = x
        
        #2
        x = self.encoder_conv_2(x)                                          
        x = self.encoder_bn_2(x)                                                                    
        x = self.activation(x)                                                 
        x = self._3d_pool(x)
        x = self.dropout(x)
        skip2 = x
        
        #3
        x = self.encoder_conv_3(x)                                          
        x = self.encoder_bn_3(x)                                                                    
        x = self.activation(x)                                                 
        x = self._3d_pool(x)
        x = self.dropout(x)

        #3
        x = self._3d_upsample(x)
        x = self.decoder_conv_3(x)                                          
        x = self.decoder_bn_3(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = x + skip2

        #2
        x = self._3d_upsample(x)
        x = self.decoder_conv_2(x)                                          
        x = self.decoder_bn_2(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = x + skip1

        #1
        x = self._3d_upsample(x)
        x = self.decoder_conv_1(x)  
        x = torch.mean(x,axis=2,keepdims=True)                                      
        
        return x
    
class UD_LSTM(nn.Module):
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
    
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, config):
        super(ConvLSTM, self).__init__()

        config = config['arch']['params']

        self._check_kernel_size_consistency(eval(config['kernel_size']))

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(eval(config['kernel_size']), config['num_layers'])
        hidden_dim = self._extend_for_multilayer(config['hidden_dim'], config['num_layers'])
        if not len(kernel_size) == len(hidden_dim) == config['num_layers']:
            raise ValueError('Inconsistent list length.')

        self.input_dim = config['input_dim']
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = config['num_layers']
        self.batch_first = config['batch_first']
        self.bias = config['bias']
        self.return_all_layers = config['return_all_layers']

        self.final_conv = config['final_conv']
        if self.final_conv:
            self.convolution = nn.Conv3d(config['hidden_dim'], config['output_dim'], (3,3,3), padding=1)

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        # (b, c, t, h, w) -> (b, t, c, h, w)
        input_tensor = input_tensor.permute(0, 2, 1, 3, 4)

        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1]
            last_state_list = last_state_list[-1:]

        # (b, t, c, h, w) -> (b, c, t, h, w)
        layer_output_list = layer_output_list.permute(0, 2, 1, 3, 4)

        if self.final_conv:
            layer_output_list = self.convolution(layer_output_list)

        return layer_output_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param