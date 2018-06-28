'''
Generate list of images
'''

import sys
sys.path.insert(0, '../util')
import platform_config as pc

import os, glob
from os.path import join

data_dir = pc.data_dir
work_dir = join(data_dir, 'Exp/CARLA_gen18')

n_frame = 30
frames = ['%08d.png' % i for i in range(1, n_frame+1, 10)]


seqs = glob.glob(join(work_dir, 'e*'))
seqs = sorted(list(filter(lambda s: os.path.isdir(s), seqs)))

with open(join(work_dir, 'list-rgb-0.1.txt'), 'w') as f_rgb:
    for seq in seqs:
        seq_name = os.path.basename(seq)
        for frame in frames:
            f_rgb.write(seq_name + '/RGB/' + frame +'\n')

