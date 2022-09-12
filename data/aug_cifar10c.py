import numpy as np
import torchvision.transforms as tform
from .util import decode_input

from .third_party.imagenetc.synpert_cifar10c import gaussian_noise, shot_noise, impulse_noise, speckle_noise
from .third_party.imagenetc.synpert_cifar10c import gaussian_blur, glass_blur, defocus_blur, motion_blur, zoom_blur
from .third_party.imagenetc.synpert_cifar10c import fog, frost, snow
from .third_party.imagenetc.synpert_cifar10c import spatter, contrast, brightness, saturate, jpeg_compression, pixelate, elastic_transform


# 1. noise 1/4
class GaussianNoise:
    def __init__(self, severity):
        self.severity = severity

    def __call__(self, img):
        img, label = decode_input(img)
        if self.severity > 0:
            img = gaussian_noise(img, self.severity)
            ## convert type
            img = np.array(img).astype(np.uint8)
            ## maintain origial PIL format
            img = tform.ToPILImage()(img)
        return (img, label)

    def __repr__(self):
        return f'GaussianNoise(severity={self.severity})'


# 2. noise 2/4 
class ShotNoise:
    def __init__(self, severity):
        self.severity = severity

    def __call__(self, img):
        img, label = decode_input(img)
        if self.severity > 0:
            img = shot_noise(img, self.severity)
            ## convert type
            img = np.array(img).astype(np.uint8)
            ## maintain origial PIL format
            img = tform.ToPILImage()(img)
        return (img, label)

    def __repr__(self):
        return f'ShotNoise(severity={self.severity})'

# 3. noise 3/4 
class ImpulseNoise:
    def __init__(self, severity):
        self.severity = severity

    def __call__(self, img):
        img, label = decode_input(img)
        if self.severity > 0:
            img = impulse_noise(img, self.severity)
            ## convert type
            img = np.array(img).astype(np.uint8)
            ## maintain origial PIL format
            img = tform.ToPILImage()(img)
        return (img, label)

    def __repr__(self):
        return f'ImpulseNoise(severity={self.severity})'


# 15+1. noise 4/4 
class SpeckleNoise:
    def __init__(self, severity):
        self.severity = severity

    def __call__(self, img):
        img, label = decode_input(img)
        if self.severity > 0:
            img = speckle_noise(img, self.severity)
            ## convert type
            img = np.array(img).astype(np.uint8)
            ## maintain origial PIL format
            img = tform.ToPILImage()(img)
        return (img, label)

    def __repr__(self):
        return f'SpeckleNoise(severity={self.severity})'


# 15+2. blur 
class GaussianBlur:
    def __init__(self, severity):
        self.severity = severity

    def __call__(self, img):
        img, label = decode_input(img)
        if self.severity > 0:
            img = gaussian_blur(img, self.severity)
            ## convert type
            img = np.array(img).astype(np.uint8)
            ## maintain origial PIL format
            img = tform.ToPILImage()(img)
        return (img, label)

    def __repr__(self):
        return f'GaussianBlur(severity={self.severity})'


# 4. blur (this is slow)
class GlassBlur:
    def __init__(self, severity):
        self.severity = severity

    def __call__(self, img):
        img, label = decode_input(img)
        if self.severity > 0:
            img = glass_blur(img, self.severity)
            ## convert type
            img = np.array(img).astype(np.uint8)
            ## maintain origial PIL format
            img = tform.ToPILImage()(img)
        return (img, label)

    def __repr__(self):
        return f'GlassBlur(severity={self.severity})'

    
# 5. blur 
class DefocusBlur:
    def __init__(self, severity):
        self.severity = severity

    def __call__(self, img):
        img, label = decode_input(img)
        if self.severity > 0:
            img = defocus_blur(img, self.severity)
            ## convert type
            img = np.array(img).astype(np.uint8)
            ## maintain origial PIL format
            img = tform.ToPILImage()(img)
        return (img, label)

    def __repr__(self):
        return f'DefocusBlur(severity={self.severity})'

    
# 6. blur 
class MotionBlur:
    def __init__(self, severity):
        self.severity = severity

    def __call__(self, img):
        img, label = decode_input(img)
        if self.severity > 0:
            img = motion_blur(img, self.severity)
            ## convert type
            img = np.array(img).astype(np.uint8)
            ## maintain origial PIL format
            img = tform.ToPILImage()(img)
        return (img, label)

    def __repr__(self):
        return f'MotionBlur(severity={self.severity})'

    
# 7. blur 
class ZoomBlur:
    def __init__(self, severity):
        self.severity = severity

    def __call__(self, img):
        img, label = decode_input(img)
        if self.severity > 0:
            img = zoom_blur(img, self.severity)
            ## convert type
            img = np.array(img).astype(np.uint8)
            ## maintain origial PIL format
            img = tform.ToPILImage()(img)
        return (img, label)

    def __repr__(self):
        return f'ZoomBlur(severity={self.severity})'

    
# 8. weather
class Fog:
    def __init__(self, severity):
        self.severity = severity

    def __call__(self, img):
        img, label = decode_input(img)
        if self.severity > 0:
            img = fog(img, self.severity)
            ## convert type
            img = np.array(img).astype(np.uint8)
            ## maintain origial PIL format
            img = tform.ToPILImage()(img)
        return (img, label)

    def __repr__(self):
        return f'Fog(severity={self.severity})'

    
# 9. weather
class Frost:
    def __init__(self, severity):
        self.severity = severity

    def __call__(self, img):
        img, label = decode_input(img)
        if self.severity > 0:
            img = frost(img, self.severity)
            ## convert type
            img = np.array(img).astype(np.uint8)
            ## maintain origial PIL format
            img = tform.ToPILImage()(img)
        return (img, label)

    def __repr__(self):
        return f'Frost(severity={self.severity})'


# 10. weather (this is slow)
class Snow:
    def __init__(self, severity):
        self.severity = severity

    def __call__(self, img):
        img, label = decode_input(img)
        if self.severity > 0:
            img = snow(img, self.severity)
            ## convert type
            img = np.array(img).astype(np.uint8)
            ## maintain origial PIL format
            img = tform.ToPILImage()(img)
        return (img, label)

    def __repr__(self):
        return f'Snow(severity={self.severity})'


# 11. digital
class Contrast:
    def __init__(self, severity):
        self.severity = severity

    def __call__(self, img):
        img, label = decode_input(img)
        if self.severity > 0:
            img = contrast(img, self.severity)
            ## convert type
            img = np.array(img).astype(np.uint8)
            ## maintain origial PIL format
            img = tform.ToPILImage()(img)
        return (img, label)

    def __repr__(self):
        return f'Contrast(severity={self.severity})'


# 12. digital
class Brightness:
    def __init__(self, severity):
        self.severity = severity

    def __call__(self, img):
        img, label = decode_input(img)
        if self.severity > 0:
            img = brightness(img, self.severity)
            ## convert type
            img = np.array(img).astype(np.uint8)
            ## maintain origial PIL format
            img = tform.ToPILImage()(img)
        return (img, label)

    def __repr__(self):
        return f'Brightness(severity={self.severity})'
    


# 13. digital
class JPEGCompression:
    def __init__(self, severity):
        self.severity = severity

    def __call__(self, img):
        img, label = decode_input(img)
        if self.severity > 0:
            img = jpeg_compression(img, self.severity)
            ## convert type
            img = np.array(img).astype(np.uint8)
            ## maintain origial PIL format
            img = tform.ToPILImage()(img)
        return (img, label)

    def __repr__(self):
        return f'JPEGCompression(severity={self.severity})'

    
# 14. digital
class Pixelate:
    def __init__(self, severity):
        self.severity = severity

    def __call__(self, img):
        img, label = decode_input(img)
        if self.severity > 0:
            img = pixelate(img, self.severity)
            ## convert type
            img = np.array(img).astype(np.uint8)
            ## maintain origial PIL format
            img = tform.ToPILImage()(img)
        return (img, label)

    def __repr__(self):
        return f'Pixelate(severity={self.severity})'

# 15. digital
class Elastic:
    def __init__(self, severity):
        self.severity = severity

    def __call__(self, img):
        img, label = decode_input(img)
        if self.severity > 0:
            img = elastic_transform(img, self.severity)
            ## convert type
            img = np.array(img).astype(np.uint8)
            ## maintain origial PIL format
            img = tform.ToPILImage()(img)
        return (img, label)

    def __repr__(self):
        return f'Elastic(severity={self.severity})'
