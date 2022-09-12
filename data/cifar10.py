import os, sys
import glob
import time
import random
import pickle
import numpy as np
from PIL import Image

import torch as tc
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets

#from torchvision.datasets.utils import check_integrity
import torchvision.transforms as tforms
import data
import data.custom_transforms as ctforms


# def _load_meta_file(meta_file):
#     if check_integrity(meta_file):
#         return tc.load(meta_file)
#     else:
#         raise RuntimeError("Meta file not found or corrupted.")

    
# def label_to_name(root, label_to_wnid):
#     meta_file = os.path.join(root, 'meta.bin')
#     wnid_to_names = _load_meta_file(meta_file)[0]

#     names = [wnid_to_names[wnid][0].replace(' ', '_').replace('\'', '_') for wnid in label_to_wnid]
#     return names

class CIFAR10Dataset:
    def __init__(self, x, y, classes, class_to_idx, transform):
        self.x = x
        self.y = y
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform


    def __len__(self):
        return len(self.y)

    
    def __getitem__(self, index):
        sample, target = self.x[index], self.y[index]
        sample = Image.fromarray(sample)

        if self.transform is not None:
            sample, target = self.transform((sample, target))
        return sample, target
        

class CIFAR10:
    def __init__(
            self, root, batch_size,
            aug_types=[],
            sample_size={'train': None, 'val': None, 'test': None},
            seed=None,
            num_workers=4,
            #ext='JPEG',
    ):
        
        ## data augmentation
        tforms_aug = data.get_aug_tforms(aug_types)

        ## default transforms
        tforms_dft_rnd = [
            ctforms.RandomCrop(32, padding=4),
            ctforms.RandomHorizontalFlip(),
            *tforms_aug, # add before ToTensor()
            ctforms.ToTensor(),
            ctforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
        ]
        tforms_dft = [
            #ctforms.RandomCrop(32, padding=4),
            #ctforms.RandomHorizontalFlip(),
            *tforms_aug, # add before ToTensor()
            ctforms.ToTensor(),
            ctforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
        ]
        


        ## transformations for each data split
        tforms_train = tforms_dft_rnd 
        tforms_val = tforms_dft #tforms_dft_rnd 
        tforms_test = tforms_dft #tforms_dft_rnd
        print("[tforms_train] ", tforms_train)
        print("[tforms_val] ", tforms_val)
        print("[tforms_test] ", tforms_test)


        ## load data using pytorch datasets
        train_ds = datasets.CIFAR10(root=root, train=True, download=True, transform=None)
        test_ds = datasets.CIFAR10(root=root, train=False, download=True, transform=None)

        ## get splits
        x_test, y_test = np.array(test_ds.data), np.array(test_ds.targets)
        
        index_rnd = data.get_random_index(len(x_test), len(x_test), seed)
        index_val = index_rnd[:len(index_rnd)//2]
        index_test = index_rnd[len(index_rnd)//2:]

        x_train, y_train = train_ds.data, train_ds.targets
        x_val, y_val = x_test[index_val], y_test[index_val]
        x_test, y_test = x_test[index_test], y_test[index_test]
               
        ## get class name
        classes, class_to_idx = train_ds.classes, train_ds.class_to_idx

        ## create a data loader for training
        ds = Subset(CIFAR10Dataset(x_train, y_train, classes, class_to_idx, transform=tforms.Compose(tforms_train)),
                    data.get_random_index(len(y_train),
                                          len(y_train) if sample_size['train'] is None else sample_size['train'], seed))
        self.train = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        ## create a data loader for validation
        ds = Subset(CIFAR10Dataset(x_val, y_val, classes, class_to_idx, transform=tforms.Compose(tforms_val)),
                    data.get_random_index(len(y_val),
                                          len(y_val) if sample_size['val'] is None else sample_size['val'], seed))
        self.val = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        ## create a data loader for test
        ds = Subset(CIFAR10Dataset(x_test, y_test, classes, class_to_idx, transform=tforms.Compose(tforms_test)),
                    data.get_random_index(len(y_test),
                                          len(y_test) if sample_size['test'] is None else sample_size['test'], seed))
        self.test = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        
        # ## add id-name map
        # self.names = label_to_name(root, self.test.dataset.dataset.classes)

        ## print data statistics
        print(f'#train = {len(self.train.dataset)}, #val = {len(self.val.dataset)}, #test = {len(self.test.dataset)}')

        
if __name__ == '__main__':
    dsld = data.CIFAR10(root='data/CIFAR10', batch_size=100, sample_size={'train': None, 'val': None, 'test': None})

    
## CIFAR10
#train =  50000
#val =  5000
#test =  5000



