import os, sys
import glob
import time
import random
import pickle
import numpy as np
import os.path as osp
import types
import json

import torch
import torch as tc
from torch.utils.data import Dataset, DataLoader

from .third_party.fewshot_re_kit.sentence_encoder import CNNSentenceEncoder, BERTSentenceEncoder, BERTPAIRSentenceEncoder, RobertaSentenceEncoder, RobertaPAIRSentenceEncoder

#import data

'''
code from https://github.com/thunlp/FewRel
'''
class FewRelDataset(Dataset):
    """
    FewRel Dataset
    """
    def __init__(self, name, encoder, n_datasets, N, K, Q, na_rate, root, seed):
        self.root = root
        path = os.path.join(root, name + ".json")
        if not os.path.exists(path):
            print(f"[ERROR] Data file does not exist: {path}")
            assert(0)
        self.json_data = json.load(open(path))
        self.classes = list(self.json_data.keys())
        self.n_datasets = n_datasets
        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate
        self.encoder = encoder
        random.seed(seed)
        self.seed_list = [random.randint(0, 1e6) for _ in range(self.n_datasets)]
        random.seed(time.time())
        
        
    def __getraw__(self, item):
        word, pos1, pos2, mask = self.encoder.tokenize(item['tokens'],
            item['h'][2][0],
            item['t'][2][0])
        return word, pos1, pos2, mask 

    
    def __additem__(self, d, word, pos1, pos2, mask):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)

        
    def __getitem__(self, index):
        random.seed(self.seed_list[index])
        
        target_classes = random.sample(self.classes, self.N)
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [] }
        query_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [] }
        query_label = []
        Q_na = int(self.na_rate * self.Q)
        na_classes = list(filter(lambda x: x not in target_classes,  
            self.classes))

        for i, class_name in enumerate(target_classes):
            indices = np.random.choice(
                    list(range(len(self.json_data[class_name]))), 
                    self.K + self.Q, False)
            count = 0
            for j in indices:
                word, pos1, pos2, mask = self.__getraw__(
                        self.json_data[class_name][j])
                word = torch.tensor(word).long()
                pos1 = torch.tensor(pos1).long()
                pos2 = torch.tensor(pos2).long()
                mask = torch.tensor(mask).long()
                if count < self.K:
                    self.__additem__(support_set, word, pos1, pos2, mask)
                else:
                    self.__additem__(query_set, word, pos1, pos2, mask)
                count += 1

            query_label += [i] * self.Q

        # NA
        for j in range(Q_na):
            cur_class = np.random.choice(na_classes, 1, False)[0]
            index = np.random.choice(
                    list(range(len(self.json_data[cur_class]))),
                    1, False)[0]
            word, pos1, pos2, mask = self.__getraw__(
                    self.json_data[cur_class][index])
            word = torch.tensor(word).long()
            pos1 = torch.tensor(pos1).long()
            pos2 = torch.tensor(pos2).long()
            mask = torch.tensor(mask).long()
            self.__additem__(query_set, word, pos1, pos2, mask)
        query_label += [self.N] * Q_na

        random.seed(time.time())

        return support_set, query_set, query_label

    
    def __len__(self):
        #return 1000000000
        return self.n_datasets

    
def collate_fn(data):
    batch_support = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_query = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_label = []
    support_sets, query_sets, query_labels = zip(*data)

    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
        for k in query_sets[i]:
            batch_query[k] += query_sets[i][k]
        batch_label += query_labels[i]
    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)
    for k in batch_query:
        batch_query[k] = torch.stack(batch_query[k], 0)
    batch_label = torch.tensor(batch_label)
    return (batch_support, batch_query), batch_label


# def get_loader(name, encoder, N, K, Q, batch_size, 
#         num_workers=8, collate_fn=collate_fn, na_rate=0, root='./data'):
#     dataset = FewRelDataset(name, encoder, N, K, Q, na_rate, root)
#     data_loader = data.DataLoader(dataset=dataset,
#             batch_size=batch_size,
#             shuffle=False,
#             pin_memory=True,
#             num_workers=num_workers,
#             collate_fn=collate_fn)
#     return iter(data_loader)


            
    
class FewRel:
    def __init__(self, root, args):
        
        seed = args.seed
        n_datasets_train = args.n_datasets_train
        n_datasets_cal = args.n_datasets_cal
        n_datasets_test = args.n_datasets_test
        num_workers = args.n_workers
        n_ways = args.n_ways
        n_shots_adapt = args.n_shots_adapt
        n_shots_cal = args.n_shots_cal
        n_shots_test = args.n_shots_test
        encoder_name = args.encoder_name

        if encoder_name == 'cnn':
            glove_mat = np.load(os.path.join(root, 'pretrain/glove/glove_mat.npy'))
            glove_word2id = json.load(open(os.path.join(root, 'pretrain/glove/glove_word2id.json')))
            max_length = args.max_length
            encoder = CNNSentenceEncoder(
                glove_mat,
                glove_word2id,
                max_length)
        else:
            raise NotImplementedError


        # train
        ds = FewRelDataset('train_wiki', encoder, n_datasets_train, n_ways, n_shots_adapt, n_shots_test, 0, root, seed=seed)
        self.train = DataLoader(dataset=ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)

        ds = FewRelDataset('val_wiki', encoder, n_datasets_cal, n_ways, n_shots_adapt, n_shots_cal, 0, root, seed=seed+1)
        self.val = DataLoader(dataset=ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)

        ds = FewRelDataset('val_wiki', encoder, n_datasets_cal, n_ways, n_shots_adapt, n_shots_cal, 0, root, seed=seed+2)
        self.cal = DataLoader(dataset=ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)

        ds = FewRelDataset('val_wiki', encoder, n_datasets_test, n_ways, n_shots_adapt, n_shots_test, 0, root, seed=seed+3)
        self.test = DataLoader(dataset=ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)

        #TODO
        self.encoder = encoder
        
        print(f'#train dataset = {len(self.train.dataset)}, '\
              f'#val datasets = {len(self.val.dataset)}, '\
              f'#cal datasets = {len(self.cal.dataset)}, '\
              f'#test datasets = {len(self.test.dataset)}')

        
if __name__ == '__main__':
    dsld = FewRel(root='data/fewrel',
                        args=types.SimpleNamespace(n_datasets_train=200, n_datasets_val=100, n_datasets_cal=100, n_datasets_test=100,
                                                   n_ways=5, n_shots_adapt=5, n_shots_cal=10, n_shots_test=20, seed=0, n_workers=8))
                        

    for i, (x, y) in enumerate(dsld.train):
        print(i, x.shape, y.shape)
        print(y)
    
#train dataset = 38400, #val datasets = 9600, #test datasets = 12000



