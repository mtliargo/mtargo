'''
carla seg to color
set ego-vecicle to None (as done in SIMS)

'''

import os, glob
import numpy as np
from os.path import join
from PIL import Image
from cityscapes import c2clabel 

data_dir = '/home/mli/Data'
in_dir = join(data_dir, 'exp/CARLA_gen17/e000001/Seg')
out_dir = join(data_dir, 'exp/CARLA_gen17/e000001/SegColor')
mask_file = join(data_dir, 'exp/CARLA_gen17/ego-vehicle.png')

if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

mask = np.array(Image.open(mask_file), dtype=bool)

files = glob.glob(join(in_dir, '*.png'))
for f in files:
    img = np.array(Image.open(f))
    img = img[:, :, 0]
    img[mask] = 13
    color_seg = c2clabel(img)
    Image.fromarray(color_seg).save(join(out_dir, os.path.basename(f)))
