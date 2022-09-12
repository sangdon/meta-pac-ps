import os, sys
import numpy as np
import time
import glob
import random
import math
#from typing import Any, Callable, cast, Dict, List, Optional, Tuple
#from PIL import Image
import pickle
import warnings

import torch as tc
from torchvision.datasets.folder import default_loader


def decode_input(x):
    if type(x) is tuple:
        ## assume (img, label) tupble
        img = x[0]
        label = x[1]
    else:
        img = x
        label = None
    return img, label




"""
Simple wrapper functions
"""
def compute_num_exs(ld, verbose=False):
    n = 0
    t = time.time()
    for x, _ in ld:
        n += x.shape[0]
        if verbose:
            print("[%f sec.] n = %d"%(time.time()-t, n))
            t = time.time()
    return n


"""
functions/classes for data loaders
"""
    
def shuffle_list(list_ori, seed):
    random.seed(seed)
    random.shuffle(list_ori)
    random.seed(int(time.time()))
    return list_ori


def load_data_splits(root, ext, seed):
    
    fns_train = glob.glob(os.path.join(root, 'train', '**', '**.'+ext))
    fns_val = glob.glob(os.path.join(root, 'val', '**', '**.'+ext))
    fns_test = glob.glob(os.path.join(root, 'test', '**', '**.'+ext))
    
    ## shuffle
    fns_train = shuffle_list(fns_train, seed)
    fns_val = shuffle_list(fns_val, seed)
    fns_test = shuffle_list(fns_test, seed)

    return {'train': fns_train, 'val': fns_val, 'test': fns_test}


def get_random_index(n_ori, n, seed):
    random.seed(seed)
    if n_ori < n:
        index = [random.randint(0, n_ori-1) for _ in range(n)]
    else:
        index = [i for i in range(n_ori)]
        random.shuffle(index)
        index = index[:n]
        
    random.seed(time.time())
    return index



def find_classes(root):
    classes = [d.name for s in ['train', 'val', 'test'] for d in os.scandir(os.path.join(root, s)) if d.is_dir()]
    classes = list(set(classes))
    classes.sort()
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def get_class_name(fn):
    return fn.split('/')[-2]


def make_dataset(fn_list, class_to_idx):
    instances = []
    for fn in fn_list:
        class_idx = class_to_idx[get_class_name(fn)]
        item = fn, class_idx
        instances.append(item)
    return instances

    
class ImageListDataset:
    def __init__(self, fn_list, classes, class_to_idx, transform=None, loader=default_loader, return_path=False):
        self.loader = loader
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.samples = make_dataset(fn_list, class_to_idx)
        self.return_path = return_path

        
    def __getitem__(self, index):
        path, target = self.samples[index]
        target = target 
        sample = self.loader(path)
        if self.transform is not None:
            sample, target = self.transform((sample, target))
        if self.return_path:
            return path, sample, target
        else:
            return sample, target


    def __len__(self):
        return len(self.samples)



class ConcatDataLoader:
    def __init__(self, ld_list):
        self.ld_list = ld_list
        assert(len(self.ld_list) > 0)
        for ld in self.ld_list:
            assert(len(ld) > 0)
        #self.index = 0


    def __iter__(self):
        self.index = 0
        random.shuffle(self.ld_list)
        self.ld = iter(self.ld_list[self.index])
        return self

    def __next__(self):
        try:
            return next(self.ld)
        except StopIteration:
            self.index = self.index + 1
            if self.index >= len(self.ld_list):
                #self.index = 0
                raise StopIteration
            else:
                self.ld = iter(self.ld_list[self.index])
                return next(self.ld) ## assume ld is not empty since it passed the assertion in __init__()
    
    
