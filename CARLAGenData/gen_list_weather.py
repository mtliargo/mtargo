'''
Generate list of images of the same weather
'''

import sys
sys.path.insert(0, '../util')
import platform_config as pc

import os, glob
from os.path import join

import numpy as np

data_dir = pc.data_dir
work_dir = join(data_dir, 'Exp/CARLA_gen18')

weather = 2 # CloudyNoon

n_frame = 30
frames = ['%08d.png' % i for i in range(2, n_frame+1, 2)]

weather_all = np.loadtxt(join(work_dir, 'weathers.txt'), int)

seqs = glob.glob(join(work_dir, 'e*'))
seqs = sorted(list(filter(lambda s: os.path.isdir(s), seqs)))

with open(join(work_dir, 'list-rgb-cloudynoon-0.5-alternate.txt'), 'w') as f_rgb:
    for w, seq in zip(weather_all, seqs):
        if w == weather:
            seq_name = os.path.basename(seq)
            for frame in frames:
                f_rgb.write(seq_name + '/RGB/' + frame +'\n')

