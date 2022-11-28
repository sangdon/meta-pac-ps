import os, sys

import torch as tc
from torch import nn

from .util import *

def reduce(loss_vec, reduction):
    if reduction == 'mean':
        return loss_vec.mean()
    elif reduction == 'sum':
        return loss_vec.sum()
    elif reduction == 'none':
        return loss_vec
    else:
        raise NotImplementedError
    

##
## classification
##
def loss_xe(x, y, model, reduction='mean', device=tc.device('cpu'), weight=None):
    x, y = to_device(x, device), to_device(y, device)
    loss_fn = nn.CrossEntropyLoss(reduction=reduction, weight=weight.to(device) if weight else None)
    loss = loss_fn(model(x)['fh'], y)
    return {'loss': loss}


def loss_01(x, y, model, reduction='mean', device=tc.device('cpu')):
    x, y = to_device(x, device), to_device(y, device)
    yh = model(x)['yh_top']
    loss_vec = (yh != y).float()
    loss = reduce(loss_vec, reduction)
    return {'loss': loss}


##
## prediction set estimation
##
def loss_set_size(x, y, mdl, reduction='mean', device=tc.device('cpu')):
    x, y = to_device(x, device), to_device(y, device)
    loss_vec = mdl.size(x, y).float()
    loss = reduce(loss_vec, reduction)
    return {'loss': loss}


def loss_set_error(x, y, mdl, reduction='mean', device=tc.device('cpu')):
    x, y = to_device(x, device), to_device(y, device)
    loss_vec = (mdl.membership(x, y) == 0).float()
    loss = reduce(loss_vec, reduction)
    return {'loss': loss}
    
         
