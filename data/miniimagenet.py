import os, sys
import glob
import time
import random
import pickle
import numpy as np
import os.path as osp
import types

import torch as tc
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image

import data

'''
code from https://github.com/yinboc/prototypical-network-pytorch/blob/master/mini_imagenet.py
'''
class MiniImageNetDataset:

    def __init__(self, root, setname):
        csv_path = osp.join(root, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(root, 'images', name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)

        self.data = data
        self.label = label

        self.transform = transforms.Compose([
            transforms.Resize(84),
            transforms.CenterCrop(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label

    
'''
code from https://github.com/yinboc/prototypical-network-pytorch/blob/master/samplers.py
'''
class CategoriesSampler:

    def __init__(self, label, n_datasets, n_cls, n_samples):
        self.n_datasets = n_datasets
        self.n_cls = n_cls # n_ways
        self.n_samples = n_samples # could be n_shots_adapt + n_shots_eval

        label = np.array(label)
        self.ind_cls = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = tc.from_numpy(ind)
            self.ind_cls.append(ind)

            
    def __len__(self):
        return self.n_datasets

    
    def __iter__(self):
        for _ in range(self.n_datasets):
            ind_batch = []
            classes = tc.randperm(len(self.ind_cls))[:self.n_cls]
            for c in classes:
                l = self.ind_cls[c]
                pos = tc.randperm(len(l))[:self.n_samples]
                ind_batch.append(l[pos])
            ind_batch = tc.stack(ind_batch).t().reshape(-1)
            yield ind_batch

            
    
class MiniImageNet:
    def __init__(self, root, args):
        
        seed = args.seed
        n_datasets_train = args.n_datasets_train
        n_datasets_cal = args.n_datasets_cal
        n_datasets_test = args.n_datasets_test
        num_workers = args.n_workers
        n_ways = args.n_ways
        
        n_samples_train = args.n_shots_adapt + args.n_shots_train
        n_samples_val = args.n_shots_adapt + args.n_shots_val
        n_samples_cal = args.n_shots_adapt + args.n_shots_cal
        n_samples_test = args.n_shots_adapt + args.n_shots_test

        def collate_fn(batch, n_queries):
            x = tc.stack([x for x, y in batch], 0)
            y = tc.arange(n_ways).long().repeat(n_queries)
            return x, y

        ds = MiniImageNetDataset(root, 'train')
        sampler = CategoriesSampler(ds.label, n_datasets_train, n_ways, n_samples_train)
        self.train = DataLoader(dataset=ds, batch_sampler=sampler, collate_fn=lambda batch: collate_fn(batch, n_samples_train-args.n_shots_adapt),
                                num_workers=num_workers, pin_memory=True)

        ds = MiniImageNetDataset(root, 'val')
        sampler = CategoriesSampler(ds.label, n_datasets_cal, n_ways, n_samples_val)
        self.val = DataLoader(dataset=ds, batch_sampler=sampler, collate_fn=lambda batch: collate_fn(batch, n_samples_val-args.n_shots_adapt),
                              num_workers=num_workers, pin_memory=True)

        ds = MiniImageNetDataset(root, 'val')
        sampler = CategoriesSampler(ds.label, n_datasets_cal, n_ways, n_samples_cal)
        self.cal = DataLoader(dataset=ds, batch_sampler=sampler, collate_fn=lambda batch: collate_fn(batch, n_samples_cal-args.n_shots_adapt),
                              num_workers=num_workers, pin_memory=True)

        ds = MiniImageNetDataset(root, 'test')
        sampler = CategoriesSampler(ds.label, n_datasets_test, n_ways, n_samples_test)
        self.test = DataLoader(ds, batch_sampler=sampler, collate_fn=lambda batch: collate_fn(batch, n_samples_test-args.n_shots_adapt),
                               num_workers=num_workers, pin_memory=True)

        print(f'#train dataset = {len(self.train.dataset)}, '\
              f'#val datasets = {len(self.val.dataset)}, '\
              f'#cal datasets = {len(self.cal.dataset)}, '\
              f'#test datasets = {len(self.test.dataset)}')

        
if __name__ == '__main__':
    dsld = MiniImageNet(root='data/miniimagenet',
                        args=types.SimpleNamespace(n_datasets=200, n_ways=5, n_shots_adapt=5, n_shots_cal=10, n_shots_test=20, seed=0, n_workers=8))
                        

    for i, (x, y) in enumerate(dsld.train):
        print(i, x.shape, y.shape)
        print(y)
    
#train dataset = 38400, #val datasets = 9600, #test datasets = 12000



