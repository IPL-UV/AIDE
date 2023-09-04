import torch.nn as nn
import torch
from torch.nn.functional import one_hot
from torch import Tensor
from typing import Union
from typing import Optional

class DiceLoss1D(nn.Module):
    'Simple DICE loss implementation for 1D data'
    def __init__(self, eps:float=1e-8, ignore_cl0:bool=True, weights:list=None, **kwargs):
        super().__init__()
        self.eps= eps
        self.ignore_cl0= ignore_cl0
        
         #(class) -> (1, class)
        self.weights= torch.Tensor(weights)[None] if weights is not None else None
        
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        'Assume the input dimensions are (batch, class, t)'
        x= x.permute(0,2,1).contiguous() #(b, c, t) -> (b, t, c)
        nc= x.shape[-1] #Number of classes
        y= one_hot(y, num_classes=nc) 
        if self.ignore_cl0:
            x, y= x[...,1:], y[...,1:]
        dsc= 2 * (x*y).sum(axis=1) / (x.sum(axis=1) + y.sum(axis=1) + self.eps)
        if self.weights is not None:
            #Make sure that no weight for cl0 is included if ignore_cl0 is set to True!
            dsc*= weights
        return 1. - dsc.mean()
    
class CrossEntropyDiceLoss1D(nn.Module):
    'Convex combination of DICE and cross entropy given lam € [0,1]'
    def __init__(self, lam:float=0.05, weights:list=None, eps:float=1e-8, ignore_cl0:bool=True, **kwargs):
        '''Note that parameters `weights`, `eps`, and `ignore_cl0` are only passed to 
           the constructor of DiceLoss1D, and hence ignored by CrossEntropyLoss'''
        super().__init__()
        self.eps= eps
        self.ingore_cl0= ignore_cl0
        self.lam= lam
        
        self.ce = nn.CrossEntropyLoss(**kwargs)
        self.dsc= DiceLoss1D(eps=eps, ignore_cl0=ignore_cl0, weights=weights)
        
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        'Assume the input dimensions are (batch, class, t)'
        return self.lam * self.ce(x, y) + (1 - self.lam) * self.dsc(x, y)