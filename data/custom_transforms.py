import os, sys
import torch as tc
from torchvision import transforms as tforms
from data import decode_input


class Identity:
    def __call__(self, x):
        return x

    def __repr__(self):
        return 'Identity()'

class CustomTransform:
    def __call__(self, img):
        img, label = decode_input(img)
        img = self.tf(img)
        return (img, label)
    
    
class ToJustTensor(CustomTransform):
    def __init__(self):
        self.tf = lambda x: tc.tensor(x)

    def __repr__(self):
        return 'ToJustTensor()'

    
class ToTensor(CustomTransform):
    def __init__(self):
        self.tf = tforms.ToTensor()

    def __repr__(self):
        return 'ToTensor()'

    
class Normalize(CustomTransform):
    def __init__(self, mean, std):
        self.tf = tforms.Normalize(mean, std)

    def __repr__(self):
        return f'Normalize(mean={self.tf.mean}, std={self.tf.std})'

    
class Grayscale(CustomTransform):
    def __init__(self, n_channels):
        self.tf = tforms.Grayscale(n_channels)

    def __repr__(self):
        return 'Grayscale()' 
        

class Resize(CustomTransform):
    def __init__(self, size):
        self.tf = tforms.Resize(size)

    def __repr__(self):
        return f'Resize(size={self.tf.size})'

    
class CenterCrop(CustomTransform):
    def __init__(self, size):
        self.tf = tforms.CenterCrop(size)

    def __repr__(self):
        return f'CenterCrop(size={self.tf.size})'

        
class RandomResizedCrop(CustomTransform):
    def __init__(self, size):
        self.tf = tforms.RandomResizedCrop(size)

    def __repr__(self):
        return f'RandomResizedCrop(size={self.tf.size})'

    
class RandomCrop(CustomTransform):
    def __init__(self, size, padding):
        self.tf = tforms.RandomCrop(size, padding=padding)

    def __repr__(self):
        return f'RandomCrop(size={self.tf.size}, padding={self.tf.padding})'

        
class RandomHorizontalFlip(CustomTransform):
    def __init__(self):
        self.tf = tforms.RandomHorizontalFlip()

    def __repr__(self):
        return f'RandomHorizontalFlip()'
