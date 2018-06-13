'''
Colorize CARLA masks without any post-processing

'''

import os, glob
import numpy as np
from os.path import join
from PIL import Image
from cityscapes import c2clabel 

data_dir = '/home/mli/Data'

in_dir = join(data_dir, 'Exp/CARLA_gen17/e000003/Seg')
out_dir = join(data_dir, 'Exp/CARLA_gen17/e000003/SegColorRaw')

if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

files = glob.glob(join(in_dir, '*.png'))
for f in files:
    img = np.array(Image.open(f))
    img = img[:, :, 0]
    color_seg = c2clabel(img)
    Image.fromarray(color_seg).save(join(out_dir, os.path.basename(f)))

## batch mode
# seqs = glob.glob(join(data_dir, 'Exp/CARLA_gen17/e*'))
# seqs = list(filter(lambda s: os.path.isdir(s), seqs))

# for s in seqs:
#     in_dir = join(s, 'Seg')
#     out_dir = join(s, 'SegColorRaw')
    
#     if not os.path.isdir(out_dir):
#         os.makedirs(out_dir)

#     files = glob.glob(join(in_dir, '*.png'))
#     for f in files:
#         img = np.array(Image.open(f))
#         img = img[:, :, 0]
#         color_seg = c2clabel(img)
#         Image.fromarray(color_seg).save(join(out_dir, os.path.basename(f)))
