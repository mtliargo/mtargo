'''
center crop
resize to 512x256
'''

import os, glob
import numpy as np
from os.path import join
from PIL import Image
from cityscapes import c2clabel 

data_dir = '/home/mli/Data'
in_dir = join(data_dir, 'carla-data/ped-50-veh-20-wea-1-pos-0-17-rep0/seg')
out_dir = join(data_dir, 'carla-data/cvt1')

if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

files = glob.glob(join(in_dir, '*.png'))
for f in files:
    img = Image.open(f)
    w, h = img.size
    # center crop to 2:1
    assert(w >= 2*h)
    left = int(w/2) - h
    right = left + 2*h
    upper = 0
    lower = h
    img = img.crop((left, upper, right, lower))
    img = img.resize((512, 256), Image.NEAREST)
    color_seg = c2clabel(np.array(img))
    Image.fromarray(color_seg).save(join(out_dir, os.path.basename(f)))
