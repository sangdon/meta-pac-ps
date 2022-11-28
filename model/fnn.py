import os, sys
import warnings
import threading

import torch as tc
import torch.nn as nn
import torch.nn.functional as F

class FNN(nn.Module):
    def __init__(self, n_in, n_out, n_hiddens, n_layers, path_pretrained=None):
        super().__init__()
        
        models = []
        for i in range(n_layers):
            n = n_in if i == 0 else n_hiddens
            models.append(nn.Linear(n, n_hiddens))
            models.append(nn.ReLU())
            models.append(nn.Dropout(0.5))
        self.feat = nn.Sequential(*models)
        self.classifier = nn.Linear(n_hiddens if n_hiddens is not None else n_in, n_out)

        
    def forward(self, x, training=False):
        if training:
            self.train()
        else:
            self.eval()
        feat = self.feat(x)
        logits = self.classifier(feat)
        #logits = self.model(x)
        if logits.shape[1] == 1:
            probs = tc.sigmoid(logits)
        else:
            probs = F.softmax(logits, -1)
        return {'fh': logits, 'ph': probs, 'yh_top': logits.argmax(-1), 'ph_top': probs.max(-1)[0], 'feat': feat}


class Linear(FNN):
    def __init__(self, n_in, n_out, n_hiddens=None, path_pretrained=None):
        super().__init__(n_in, n_out, n_hiddens, n_layers=0, path_pretrained=path_pretrained)


class SmallFNN(FNN):
    def __init__(self, n_in, n_out, n_hiddens=500, path_pretrained=None):
        super().__init__(n_in, n_out, n_hiddens, n_layers=1, path_pretrained=path_pretrained)

    
class MidFNN(FNN):
    def __init__(self, n_in, n_out, n_hiddens=500, path_pretrained=None):
        super().__init__(n_in, n_out, n_hiddens, n_layers=2, path_pretrained=path_pretrained)

        
class BigFNN(FNN):
    def __init__(self, n_in, n_out, n_hiddens=500, path_pretrained=None):
        super().__init__(n_in, n_out, n_hiddens, n_layers=4, path_pretrained=path_pretrained)


class DiabetesFNN(FNN):
    def __init__(self, n_in=385, n_out=2, n_hiddens=500, path_pretrained=None):
        super().__init__(n_in=n_in, n_out=n_out, n_hiddens=n_hiddens, n_layers=2, path_pretrained=path_pretrained)

        
class HeartFNN(FNN):
    def __init__(self, n_in=54, n_out=2, n_hiddens=100, path_pretrained=None):
        super().__init__(n_in=n_in, n_out=n_out, n_hiddens=n_hiddens, n_layers=2, path_pretrained=path_pretrained)





