import os, sys
import warnings
import numpy as np

import torch
import torch as tc
from torch import nn
import torch.nn.functional as F
from torchvision import models
import threading

import model

from data.third_party.fewshot_re_kit.sentence_encoder import CNNSentenceEncoder, BERTSentenceEncoder, BERTPAIRSentenceEncoder, RobertaSentenceEncoder, RobertaPAIRSentenceEncoder


def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits


class ProtoNetGeneral(nn.Module):
    def __init__(self, backbone, n_samples_adapt, n_ways, path_pretrained=None):
        super().__init__()
        self.backbone = getattr(model, backbone)()
        self.n_samples_adapt = n_samples_adapt
        self.n_ways = n_ways
        
        if path_pretrained is not None:
            # self.load_state_dict({k.replace('model.', '').replace('module.', '').replace('mdl.', ''): v for k, v in
            #                       tc.load(path_pretrained, map_location=tc.device('cpu')).items()})
            self.load_state_dict({k.replace('module.', '').replace('mdl.', ''): v for k, v in
                                  tc.load(path_pretrained, map_location=tc.device('cpu')).items()})


    def forward(self, x, training=False):
        if training:
            self.train()
        else:
            self.eval()

        x_adapt = x['x_adapt']
        y_adapt = x['y_adapt']
        x_eval = x['x_eval']

        query = self.backbone(x_eval, training=training)['feat']
        
        # compute proto types
        proto = self.backbone(x_adapt, training=training)['feat']
        proto_mean = []
        cls_inval = []
        for i in range(self.n_ways):
            ind = y_adapt == i
            if ind.any():
                proto_mean.append(proto[ind].mean(dim=0))
            else:
                cls_inval.append(i)
                proto_mean.append(tc.zeros(proto.shape[1], device=proto.device))
        proto_mean = tc.vstack(proto_mean)

        # compute logits
        logits = euclidean_metric(query, proto_mean)
        if len(cls_inval):
            for i in cls_inval:
                logits[i] = -1e6 # put a small value

        return {'fh': logits, 'ph': F.softmax(logits, -1), 'yh_top': logits.argmax(-1), 'ph_top': F.softmax(logits, -1).max(-1)[0]}

    
class ProtoNet(nn.Module):
    def __init__(self, backbone, n_shots, n_ways, path_pretrained=None):
        super().__init__()
        self.backbone = getattr(model, backbone)()
        self.n_shots_adapt = n_shots
        self.n_ways = n_ways
        if path_pretrained is not None:
            # self.load_state_dict({k.replace('model.', '').replace('module.', '').replace('mdl.', ''): v for k, v in
            #                       tc.load(path_pretrained, map_location=tc.device('cpu')).items()})
            self.load_state_dict({k.replace('module.', '').replace('mdl.', ''): v for k, v in
                                  tc.load(path_pretrained, map_location=tc.device('cpu')).items()})


    def forward(self, x, training=False):
        if training:
            self.train()
        else:
            self.eval()
            
        p = self.n_shots_adapt * self.n_ways
        x_shots, x_query = x[:p], x[p:]

        #print('1', x_shots.shape)
        res = self.backbone(x_shots, training=training)
        proto = res['feat']
        #print('2', proto.shape)
        #print('2. logits', res['fh'].shape)
        proto = proto.reshape(self.n_shots_adapt, self.n_ways, -1).mean(dim=0)
        query = self.backbone(x_query, training=training)['feat']

        #print(query.shape, proto.shape)
        logits = euclidean_metric(query, proto)

        return {'fh': logits, 'ph': F.softmax(logits, -1), 'yh_top': logits.argmax(-1), 'ph_top': F.softmax(logits, -1).max(-1)[0]}

    
class ProtoNetNLP(nn.Module):
    def __init__(self, encoder, n_shots_adapt, n_shots_test, n_ways, path_pretrained=None):
        super().__init__()
        self.encoder = encoder
        self.drop = nn.Dropout()
        self.n_shots_adapt = n_shots_adapt
        self.n_shots_test = n_shots_test
        self.n_ways = n_ways
        
        if path_pretrained is not None:
            self.load_state_dict({k.replace('model.', '').replace('module.', '').replace('mdl.', ''): v for k, v in
                                  tc.load(path_pretrained, map_location=tc.device('cpu')).items()})

            
    def __dist__(self, x, y, dim):
        # if self.dot:
        #     return (x * y).sum(dim)
        # else:
        #     return -(torch.pow(x - y, 2)).sum(dim)
        return -(torch.pow(x - y, 2)).sum(dim)

        
    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)
    

    def forward(self, x, training=False):
        if training:
            self.train()
        else:
            self.eval()

        support, query = x
        N = self.n_ways
        K = self.n_shots_adapt
        if 'n_ways' in query:
            Q = query['n_ways'] * query['n_shots']
        else:
            Q = self.n_shots_test*self.n_ways

        #print('protonet', query['word'].shape, Q)
        
        support_emb = self.encoder(support) # (B * N * K, D), where D is the hidden size
        query_emb = self.encoder(query) # (B * Q, D)
        hidden_size = support_emb.size(-1)
        support = self.drop(support_emb)
        query = self.drop(query_emb)
        support = support.view(-1, N, K, hidden_size) # (B, N, K, D)
        query = query.view(-1, Q, hidden_size) # (B, Q, D)

        # Prototypical Networks 
        # Ignore NA policy
        support = torch.mean(support, 2) # Calculate prototype for each class
        logits = self.__batch_dist__(support, query) # (B, total_Q, N)

        minn, _ = logits.min(-1)
        logits = torch.cat([logits, minn.unsqueeze(2) - 1], 2) # (B, total_Q, N + 1)
        logits = logits.view(-1, N+1)
        
        #logits = logits.view(-1, N)
        
        return {'fh': logits, 'ph': F.softmax(logits, -1), 'yh_top': logits.argmax(-1), 'ph_top': F.softmax(logits, -1).max(-1)[0]}
