import os, sys
import glob
import shutil

root_ori = './mini-imagenet'
os.makedirs('images')

for fn in glob.glob(os.path.join(root_ori, '**', '**', '*.jpg')):
    src = fn
    tar = os.path.join('images', os.path.basename(fn))
    shutil.copyfile(src, tar)
