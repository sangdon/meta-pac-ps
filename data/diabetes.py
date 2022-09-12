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
#from data.miniimagenet import CategoriesSampler
import data

'''
code from https://github.com/yinboc/prototypical-network-pytorch/blob/master/mini_imagenet.py
'''
class DiabetesDataset:

    def __init__(self, x, y):
        self.x = x
        self.y = y

        
    def __len__(self):
        return len(self.x)


    def __getitem__(self, i):
        x, y = self.x[i], self.y[i]
        x = tc.tensor(x).float()
        y = tc.tensor(y).long()
        return x, y

            
def load_data(root, seed=None):
    data = []
    for l in open(os.path.join(root, 'data.csv')).readlines()[1:]:
        l = l.strip()
        l = l.split(',')
        label = int(l[1])
        state = l[3].replace('"', '')
        pguid = l[4].replace('"', '')
        feat = [float(l[2])] + [float(i) for i in l[6:]]
        data.append({'label': label, 'state': state, 'pguid': pguid, 'feat': feat})

    pguid = set([d['pguid'] for d in data])
    pguid2label = {k: v for v, k in enumerate(pguid)}
    label2pguid = {k: v for k, v in enumerate(pguid)}

    stateid = set([d['state'] for d in data])
    stateid2label = {k: v for v, k in enumerate(stateid)}
    label2stateid = {k: v for k, v in enumerate(stateid)}
    
    label = np.array([d['label'] for d in data])
    label_ds = np.array([pguid2label[d['pguid']] for d in data])
    label_st = np.array([stateid2label[d['state']] for d in data])
    
    feat = np.stack([np.array(d['feat']) for d in data])
    assert(len(label) == len(label_ds) == feat.shape[0])
    
    if seed is not None:
        ind_rnd = np.arange(len(label))
        random.seed(seed)
        random.shuffle(ind_rnd)
        random.seed(time.time())
        label = label[ind_rnd]
        label_ds = label_ds[ind_rnd]
        feat = feat[ind_rnd]

    stats_hosp = {}
    for i in set(label_ds):
        stats_hosp[i] = float(np.sum(label_ds == i))

    stats_state = {}
    for i in set(label_st):
        stats_state[i] = float(np.sum(label_st == i))
    
        
    print(f'n_data = {len(label)}, n_hospitals = {len(set(label_ds))}, n_states = {len(set(label_st))}, n_feats = {feat.shape[1]}')
    vals = [v for v in stats_hosp.values()]
    print(f'[data per hospital] '\
          f'min = {np.min(vals)}, 25% quantile = {np.quantile(vals, 0.25)}, median = {np.median(vals)}, 75% quantile = {np.quantile(vals, 0.75)}, max = {np.max(vals)}, mean = {np.mean(vals)}')
    vals = [v for v in stats_state.values()]
    print(f'[data per state] '\
          f'min = {np.min(vals)}, 25% quantile = {np.quantile(vals, 0.25)}, median = {np.median(vals)}, 75% quantile = {np.quantile(vals, 0.75)}, max = {np.max(vals)}, mean = {np.mean(vals)}')
    
    return label, label_ds, feat


class HospitalSampler:

    def __init__(self, label, label_hosp, n_ways, n_hospitals, n_datasets, n_shots):
        self.n_datasets = n_datasets
        self.n_ways = n_ways
        self.n_hosp = n_hospitals
        self.n_shots = n_shots # could be n_shots_adapt + n_shots_{train,val,test}
        self.label = label
        
        label_hosp = np.array(label_hosp)
        self.ind_hosp = []
        for i in range(max(label_hosp) + 1):
            ind = np.argwhere(label_hosp == i).reshape(-1)
            ind = tc.from_numpy(ind)
            self.ind_hosp.append(ind)

            
    def __len__(self):
        return self.n_datasets

    
    def __iter__(self):
        for _ in range(self.n_datasets):
            hosps = tc.randperm(len(self.ind_hosp))[:self.n_hosp]

            # get all postive and negatives
            ind_pos = []
            ind_neg = []
            for h in hosps:
                ind_hosp = self.ind_hosp[h]

                ind_pos_hosp = np.argwhere(self.label[ind_hosp] == 1).reshape(-1)
                ind_neg_hosp = np.argwhere(self.label[ind_hosp] == 0).reshape(-1)

                ind_pos.append(ind_hosp[ind_pos_hosp])
                ind_neg.append(ind_hosp[ind_neg_hosp])
            ind_pos = tc.cat(ind_pos)
            ind_neg = tc.cat(ind_neg)
            ind = tc.cat((ind_pos, ind_neg))
            
            # randomly choose pos and neg examples
            ind_batch = ind[tc.randperm(len(ind))[:self.n_shots*self.n_ways]]
    
            # ind_pos = ind_pos[tc.randperm(len(ind_pos))[:self.n_shots]]
            # ind_neg = ind_neg[tc.randperm(len(ind_neg))[:self.n_shots]]            
            # ind_batch = tc.vstack((ind_neg, ind_pos)).t().flatten() # stack negative first due to the protonet convension
            yield ind_batch
            
            
    
class Diabetes:
    def __init__(self, root, args, split_ratio={'train': 0.5, 'val': 0.25, 'test': 0.25}):
        
        seed = args.seed
        n_datasets_train = args.n_datasets_train
        n_datasets_val = args.n_datasets_val
        n_datasets_cal = args.n_datasets_cal
        n_datasets_test = args.n_datasets_test
        num_workers = args.n_workers
        n_ways = args.n_ways
        assert(n_ways == 2)
        
        n_samples_train = args.n_shots_adapt + args.n_shots_train
        n_samples_val = args.n_shots_adapt + args.n_shots_val
        n_samples_cal = args.n_shots_adapt + args.n_shots_cal
        n_samples_test = args.n_shots_adapt + args.n_shots_test

        # split data
        label, label_ds, feat = load_data(root, seed=0) # load and shuffle data
        n_total = len(label)
        n_train = int(n_total * split_ratio['train'])
        n_val = int(n_total * split_ratio['val'])
        n_test = n_total - n_train - n_val
        label_train, label_val, label_test = np.split(label, [n_train, n_train+n_val])
        label_ds_train, label_ds_val, label_ds_test = np.split(label_ds, [n_train, n_train+n_val])
        feat_train, feat_val, feat_test = np.split(feat, [n_train, n_train+n_val])

        # # drop labels for adaptation shots
        # def drop_label(batch, n_ways, n_shots_adapt):
        #     x = tc.stack([x for x, y in batch], 0)
        #     y = tc.stack([y for x, y in batch], 0)
        #     # drop labels for adaptation shots (mostly due to the protonet convension)
        #     y = y[n_ways*n_shots_adapt:]        
        #     return x, y

        def split(batch, n_samples_adapt):
            x = tc.stack([x for x, y in batch], 0)
            y = tc.stack([y for x, y in batch], 0)

            x = {'x_adapt': x[:n_samples_adapt], 'y_adapt': y[:n_samples_adapt], 'x_eval': x[n_samples_adapt:]}
            y = y[n_samples_adapt:]
            return x, y
        
        ds = DiabetesDataset(feat_train, label_train)
        sampler = HospitalSampler(label_train, label_ds_train, n_ways, 300, n_datasets_train, n_samples_train) # consider a set of 300 random hospitals as one distribution
        self.train = DataLoader(dataset=ds, batch_sampler=sampler, collate_fn=lambda batch: split(batch, args.n_shots_adapt*n_ways), num_workers=num_workers, pin_memory=True)

        ds = DiabetesDataset(feat_val, label_val)
        sampler = HospitalSampler(label_val, label_ds_val, n_ways, 300, n_datasets_val, n_samples_val) # consider a set of 300 random hospitals as one distribution
        self.val = DataLoader(dataset=ds, batch_sampler=sampler, collate_fn=lambda batch: split(batch, args.n_shots_adapt*n_ways), num_workers=num_workers, pin_memory=True)

        ds = DiabetesDataset(feat_val, label_val)
        sampler = HospitalSampler(label_val, label_ds_val, n_ways, 300, n_datasets_cal, n_samples_cal) # consider a set of 300 random hospitals as one distribution
        self.cal = DataLoader(dataset=ds, batch_sampler=sampler, collate_fn=lambda batch: split(batch, args.n_shots_adapt*n_ways), num_workers=num_workers, pin_memory=True)

        ds = DiabetesDataset(feat_test, label_test)
        sampler = HospitalSampler(label_test, label_ds_test, n_ways, 300, n_datasets_test, n_samples_test) # consider a set of 300 random hospitals as one distribution
        self.test = DataLoader(dataset=ds, batch_sampler=sampler, collate_fn=lambda batch: split(batch, args.n_shots_adapt*n_ways), num_workers=num_workers, pin_memory=True)

        print(f'#train dataset = {len(self.train.dataset)}, '\
              f'#val datasets = {len(self.val.dataset)}, '\
              f'#cal datasets = {len(self.cal.dataset)}, '\
              f'#test datasets = {len(self.test.dataset)}')


        
if __name__ == '__main__':

    dsld = Diabetes(root='data/diabetes',
                    args=types.SimpleNamespace(n_datasets_train=100, n_datasets_val=50, n_datasets_cal=200, n_datasets_test=200,
                                               n_ways=2,
                                               n_shots_adapt=5, n_shots_train=5, n_shots_val=5, n_shots_cal=10, n_shots_test=20,
                                               seed=0, n_workers=8))
                        

    # for i, (x, y) in enumerate(dsld.train):
    #     print(i, x.shape, y.shape)
    #     print(y)
    


# n_data = 9948, n_datasets = 379, n_feats = 385
