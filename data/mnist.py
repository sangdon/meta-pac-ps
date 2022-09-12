import os, sys

from torchvision import transforms as tforms

import data
import data.custom_transforms as ctforms

IMAGE_SIZE=28

class MNIST(data.ClassificationData):
    def __init__(
            self, root, batch_size,
            image_size=IMAGE_SIZE, color=False,
            train_rnd=True, val_rnd=True, test_rnd=False,
            train_aug=False, val_aug=False, test_aug=False,
            aug_types=[],
            split_ratio={'train': None, 'val': None, 'test': None},
            sample_size={'train': None, 'val': None, 'test': None},
            domain_label=None,
            seed=0,
            num_workers=4,
    ):
        ## default tforms
        tforms_dft = [
            tforms.Grayscale(3 if color else 1),
            tforms.Resize([image_size, image_size]) if image_size!=IMAGE_SIZE else ctforms.Identity(),
            tforms.ToTensor(),
            ## adding Gaussian noise
            ctforms.AddGaussianNoise(0., 0.1)
        ]

        super().__init__(
            root=root, batch_size=batch_size,
            dataset_fn=data.ImageList,
            split_ratio=split_ratio,
            sample_size=sample_size,
            domain_label=domain_label,
            train_rnd=train_rnd, val_rnd=val_rnd, test_rnd=test_rnd,
            train_aug=train_aug, val_aug=val_aug, test_aug=test_aug,
            aug_types=aug_types,
            num_workers=num_workers,
            tforms_dft=tforms_dft, tforms_dft_rnd=tforms_dft,
            ext='png',
            seed=seed,
        )
        print(f'#train = {len(self.train.dataset)}, #val = {len(self.val.dataset)}, #test = {len(self.test.dataset)}')


# class MNIST_old(data.ImageData):
#     def __init__(
#             self, root, batch_size,
#             image_size=IMAGE_SIZE, color=False,
#             train_rnd=True, val_rnd=True, test_rnd=False,
#             train_aug=False, val_aug=False, test_aug=False,
#             aug_types=[],
#             num_workers=4,
#     ):
#         ## default tforms
#         tforms_dft = [
#             tforms.Grayscale(3 if color else 1),
#             tforms.Resize([image_size, image_size]) if image_size!=IMAGE_SIZE else ctforms.Identity(),
#             tforms.ToTensor(),
#         ]

#         super().__init__(
#             root=root, batch_size=batch_size,
#             image_size=image_size, color=color,
#             train_rnd=train_rnd, val_rnd=val_rnd, test_rnd=test_rnd,
#             train_aug=train_aug, val_aug=val_aug, test_aug=test_aug,
#             aug_types=aug_types,
#             num_workers=num_workers,
#             tforms_dft=tforms_dft, tforms_dft_rnd=tforms_dft,
#         )
        
            
if __name__ == '__main__':
    dsld = data.MNIST('data/mnist', 100, sample_size={'train': None, 'val': None, 'test': None}, split_ratio={'train': 0.25, 'val': 0.25, 'test': 0.5})
    print("#train = ", data.compute_num_exs(dsld.train))
    print("#val = ", data.compute_num_exs(dsld.val))
    print("#test = ", data.compute_num_exs(dsld.test))

"""
#train =  50000
#val =  10000
#test =  10000
"""
