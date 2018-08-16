'''
Build a dataset with train-val-test splits from the generated data from
'''

import sys; sys.path.insert(0, '../util')
from platform_config import data_dir, mkdir2

import os, glob, random, math
from os.path import join
from shutil import copy2
import numpy as np

# for converting to indexed images
from PIL import Image
from cityscapes import c2clabelfromrgb, cm_train

build_dir = mkdir2(join(data_dir, 'Exp/C20_S1')) # gen 18, weather 2, subsampling step 1
train_source_dir = join(data_dir, 'Exp/CARLA_gen20')
test_source_dir = join(data_dir, 'Exp/CARLA_gen20_town2')
rgb_dir = mkdir2(join(build_dir, 'RGB-1024'))
seg_dir = mkdir2(join(build_dir, 'Seg-1024'))

n_frame = 10000

train_seqs = glob.glob(join(train_source_dir, 'e*'))
train_seqs = sorted(list(filter(lambda s: os.path.isdir(s), train_seqs)))

test_seqs = glob.glob(join(test_source_dir, 'e*'))
test_seqs = sorted(list(filter(lambda s: os.path.isdir(s), test_seqs)))

n_val = 2
n_train = len(train_seqs) - n_val
seqs = train_seqs + test_seqs
n_seq = len(seqs)

cnts = 3*[0]
f = 3*[None]
splits = ['train', 'val', 'test']
for s in range(3):
    mkdir2(join(rgb_dir, splits[s]))
    mkdir2(join(seg_dir, splits[s]))

with open(join(build_dir, 'original-filename-train.txt'), 'w') as f[0], \
    open(join(build_dir, 'original-filename-val.txt'), 'w') as f[1], \
    open(join(build_dir, 'original-filename-test.txt'), 'w') as f[2]:
    for i in range(n_seq):
        print('Processing %d/%d - %s' % (i + 1, n_seq, seqs[i]))
        seq_name = os.path.basename(seqs[i])
        if i < n_train:
            s = 0
        elif i < n_train + n_val:
            s = 1
        else:
            s = 2
        for j in range(n_frame):
            cnts[s] += 1
            inname = '%08d.png' % (j + 1)
            outname = '%08d.png' % cnts[s]

            copy2(join(seqs[i], 'RGB', inname), join(rgb_dir, splits[s], outname))
            copy2(join(seqs[i], 'Seg', inname), join(seg_dir, splits[s], outname))

            f[s].write(seq_name + '/RGB/' + inname + '\n')