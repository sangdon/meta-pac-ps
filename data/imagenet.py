import os, sys
import glob
import time
import random
import pickle

import torch as tc
from torch.utils.data import DataLoader, Subset

from torchvision.datasets.utils import check_integrity
import torchvision.transforms as tforms
import data
import data.custom_transforms as ctforms


def _load_meta_file(meta_file):
    if check_integrity(meta_file):
        return tc.load(meta_file)
    else:
        raise RuntimeError("Meta file not found or corrupted.")

    
def label_to_name(root, label_to_wnid):
    meta_file = os.path.join(root, 'meta.bin')
    wnid_to_names = _load_meta_file(meta_file)[0]

    names = [wnid_to_names[wnid][0].replace(' ', '_').replace('\'', '_') for wnid in label_to_wnid]
    return names



class ImageNet:
    def __init__(
            self, root, batch_size,
            aug_types=[],
            sample_size={'train': None, 'val': None, 'test': None},
            seed=None,
            num_workers=4,
            ext='JPEG',
    ):
        
        ## data augmentation
        tforms_aug = data.get_aug_tforms(aug_types)

        ## default transformations
        tforms_dft_rnd = [
            ctforms.RandomResizedCrop(224),
            ctforms.RandomHorizontalFlip(),
            *tforms_aug, # add before ToTensor()
            ctforms.ToTensor(),
            ctforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]


        ## transformations for each data split
        tforms_train = tforms_dft_rnd 
        tforms_val = tforms_dft_rnd 
        tforms_test = tforms_dft_rnd
        print("[tforms_train] ", tforms_train)
        print("[tforms_val] ", tforms_val)
        print("[tforms_test] ", tforms_test)

        ## get class name
        classes, class_to_idx = data.find_classes(root)

        ## get splits
        split_list = data.load_data_splits(root, ext, seed)

        ## create a data loader for training
        ds = Subset(data.ImageListDataset(split_list['train'], classes, class_to_idx, transform=tforms.Compose(tforms_train)),
                    data.get_random_index(len(split_list['train']),
                                          len(split_list['train']) if sample_size['train'] is None else sample_size['train'], seed))
        self.train = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        ## create a data loader for validation
        ds = Subset(data.ImageListDataset(split_list['val'], classes, class_to_idx, transform=tforms.Compose(tforms_train)),
                    data.get_random_index(len(split_list['val']),
                                          len(split_list['val']) if sample_size['val'] is None else sample_size['val'], seed))
        self.val = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        ## create a data loader for test
        ds = Subset(data.ImageListDataset(split_list['test'], classes, class_to_idx, transform=tforms.Compose(tforms_train)),
                    data.get_random_index(len(split_list['test']),
                                          len(split_list['test']) if sample_size['test'] is None else sample_size['test'], seed))
        self.test = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        
        ## add id-name map
        self.names = label_to_name(root, self.test.dataset.dataset.classes)

        ## print data statistics
        print(f'#train = {len(self.train.dataset)}, #val = {len(self.val.dataset)}, #test = {len(self.test.dataset)}')

        
if __name__ == '__main__':
    dsld = data.ImageNet(root='imagenet', batch_size=100, sample_size={'train': None, 'val': None, 'test': None})

## ImageNet
#train =  1281167
#val =  25000
#test =  25000



