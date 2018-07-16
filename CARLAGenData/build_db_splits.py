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

# assuming the same weather
weather = 2 # CloudyNoon

source_dir = join(data_dir, 'Exp/CARLA_gen18')
build_dir = mkdir2(join(data_dir, 'Exp/C18_W2_S1')) # gen 18, weather 2, subsampling step 1
rgb_dir = mkdir2(join(build_dir, 'RGB-1024'))
seg_dir = mkdir2(join(build_dir, 'Seg-1024'))

n_start_spots = 152
n_frame = 30
frames = ['%08d.png' % i for i in range(1, n_frame+1)]
# frames = ['%08d.png' % i for i in range(2, n_frame+1, 2)]

weather_all = np.loadtxt(join(source_dir, 'weathers.txt'), int)
start_spots_all = np.loadtxt(join(source_dir, 'start-spots.txt'), int)

train_percent = 0.8
val_percent = 0.1

random.seed(0)
ss_list = list(range(n_start_spots))
random.shuffle(ss_list)
bar1 = math.ceil(n_start_spots*train_percent)
bar2 = math.ceil(n_start_spots*(train_percent + val_percent))
ss_train = ss_list[:bar1]
ss_val = ss_list[bar1:bar2]
ss_test = ss_list[bar2:]

seqs = glob.glob(join(source_dir, 'e*'))
seqs = sorted(list(filter(lambda s: os.path.isdir(s), seqs)))

n_seq = len(seqs)

cnts = 3*[0]
f = 3*[None]
splits = ['train', 'val', 'test']
for s in range(3):
    mkdir2(join(rgb_dir, splits[s]))
    mkdir2(join(seg_dir, splits[s]))

# for converting to indexed images
cm = list(cm_train.flatten())

with open(join(build_dir, 'original-filename-train.txt'), 'w') as f[0], \
    open(join(build_dir, 'original-filename-val.txt'), 'w') as f[1], \
    open(join(build_dir, 'original-filename-test.txt'), 'w') as f[2]:
    for i in range(n_seq):
        print('Processing %d/%d - %s' % (i, n_seq, seqs[i]))
        if weather_all[i] == weather:
            seq_name = os.path.basename(seqs[i])
            if start_spots_all[i] in ss_train:
                s = 0
            elif start_spots_all[i] in ss_val:
                s = 1
            else:
                s = 2
            for frame in frames:
                cnts[s] += 1
                outname = '%08d.png' % cnts[s]

                copy2(join(seqs[i], 'RGB', frame), join(rgb_dir, splits[s], outname))
                seg_color = Image.open(join(seqs[i], 'SegColor', frame))
                seg = Image.fromarray(c2clabelfromrgb(seg_color))
                seg.putpalette(cm)
                seg.save(join(seg_dir, splits[s], outname))

                # a = np.array(seg.convert('RGB'))
                # b = np.array(seg_color)
                # assert np.array_equal(a, b)

                f[s].write(seq_name + '/RGB/' + frame + '\n')